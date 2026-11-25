import { NextRequest, NextResponse } from "next/server"
import { llmJson, resolveLlmConfigs, type LlmConfig } from "@/lib/llm"

export const runtime = "nodejs"

type FactCheckClaim = {
  claim: string
  verdict: "SUPPORTED" | "CONTRADICTED" | "NOT_ENOUGH_INFO"
  confidence: number
  citations: {
    sourceIndex: number
    url: string
    quote: string
    verificationScope?: "search_snippet" | "full_page"
  }[]
  note?: string
}

type FactCheckResponse = {
  provider?: string
  model?: string
  plausibilityOnly?: boolean
  overall?: {
    truthPercent: number
    fakePercent: number
    confidence: number
    verdict: "LIKELY_TRUE" | "LIKELY_FALSE" | "MIXED" | "NOT_ENOUGH_INFO"
    summary: string
  }
  analysis?: string
  redFlags?: string[]
  summary: {
    supported: number
    contradicted: number
    notEnoughInfo: number
    evidenceScore: number
  }
  claims: FactCheckClaim[]
  sourcesUsed: { url: string; title?: string }[]
}

type SearchHit = { url: string; title?: string; snippet?: string; publisher?: string }

type SearchPlan = {
  claims: string[]
  queriesUk: string[]
  queriesEn: string[]
  entities: string[]
}

const UK_STOP = new Set(
  "і та або але що це як для про від до на по зі із не є був була було були цей ця це ті той та той вже ще дуже також або якщо щоб про теж був буде був би".split(
    " ",
  ),
)
const EN_STOP = new Set(
  "the a an and or but that this for from with into onto about after before over under not is are was were be been being it its of to in on at by as".split(
    " ",
  ),
)

function tokenize(text: string): string[] {
  return (text || "")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((w) => w.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""))
    .filter(Boolean)
}

function extractKeywordQuery(text: string, maxTerms = 8): string {
  const scored: { w: string; score: number }[] = []
  const seen = new Set<string>()
  for (const raw of tokenize(text)) {
    const lower = raw.toLowerCase()
    if (lower.length < 3) continue
    if (UK_STOP.has(lower) || EN_STOP.has(lower)) continue
    if (seen.has(lower)) continue
    seen.add(lower)
    let score = Math.min(raw.length, 12)
    if (/^[A-ZА-ЯІЇЄҐ][\p{L}]+$/u.test(raw)) score += 8 
    if (/^[A-Z]{2,}$/.test(raw)) score += 12 
    if (/^[A-Za-z][\w.-]*$/.test(raw) && /[A-Za-z]/.test(raw)) score += 6 
    if (/^\d{4}$/.test(raw)) score += 5 
    if (/\d/.test(raw) && /[A-Za-zА-Яа-яІіЇїЄєҐґ]/.test(raw)) score += 7 
    scored.push({ w: raw, score })
  }
  scored.sort((a, b) => b.score - a.score)
  return scored
    .slice(0, maxTerms)
    .map((x) => x.w)
    .join(" ")
    .trim()
}

function extractLatinQuery(text: string, maxTerms = 6): string {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of tokenize(text)) {
    if (!/[A-Za-z]/.test(raw)) continue
    const key = raw.toLowerCase()
    if (EN_STOP.has(key) || seen.has(key)) continue
    if (raw.length < 2) continue
    seen.add(key)
    out.push(raw)
    if (out.length >= maxTerms) break
  }
  return out.join(" ").trim()
}

function mapNewsApiArticles(data: any): SearchHit[] {
      const arts = Array.isArray(data?.articles) ? data.articles : []
  return arts
        .map((a: any) => ({
          url: String(a?.url ?? "").trim(),
          title: String(a?.title ?? "").trim(),
          snippet: String(a?.description ?? a?.content ?? "").trim(),
          publisher: String(a?.source?.name ?? "").trim(),
        }))
    .filter((a: SearchHit) => a.url.startsWith("http"))
}

async function newsApiFetch(params: Record<string, string>, pageSize: number): Promise<SearchHit[]> {
  const key = (process.env.NEWS_API_KEY ?? "").trim()
  if (!key) return []
  try {
    const url = new URL("https://newsapi.org/v2/everything")
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v)
    url.searchParams.set("sortBy", "relevancy")
    url.searchParams.set("pageSize", String(Math.min(10, Math.max(1, pageSize))))
    const r = await fetch(url.toString(), {
      headers: { "X-Api-Key": key },
      signal: AbortSignal.timeout(8_000),
    })
    if (!r.ok) return []
    return mapNewsApiArticles(await r.json())
    } catch {
    return []
  }
}

async function googleNewsRssSearch(
  query: string,
  locale: { hl: string; gl: string; ceid: string },
  pageSize = 5,
): Promise<SearchHit[]> {
  if (!query.trim()) return []
  try {
    const url = new URL("https://news.google.com/rss/search")
    url.searchParams.set("q", query)
    url.searchParams.set("hl", locale.hl)
    url.searchParams.set("gl", locale.gl)
    url.searchParams.set("ceid", locale.ceid)
    const r = await fetch(url.toString(), {
      headers: { "User-Agent": "DyplomFactcheck/1.0" },
      signal: AbortSignal.timeout(8_000),
    })
    if (!r.ok) return []
    const xml = await r.text()
    const out: SearchHit[] = []
    const itemRe = /<item>([\s\S]*?)<\/item>/gi
    let m: RegExpExecArray | null
    while ((m = itemRe.exec(xml)) && out.length < pageSize) {
      const block = m[1]
      const title = block.match(/<title>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/title>/i)?.[1]?.trim() ?? ""
      const link = block.match(/<link>([\s\S]*?)<\/link>/i)?.[1]?.trim() ?? ""
      const source = block.match(/<source[^>]*>([\s\S]*?)<\/source>/i)?.[1]?.trim() ?? "Google News"
      if (link.startsWith("http") && title) {
        out.push({ url: link, title, snippet: title, publisher: source })
      }
    }
    return out
  } catch {
    return []
  }
}

function uniqQueries(queries: string[], limit: number): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const q of queries) {
    const t = q.replace(/\s+/g, " ").trim()
    if (t.length < 4) continue
    const key = t.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(t)
    if (out.length >= limit) break
  }
  return out
}

async function collectEvidenceHits(
  text: string,
  plan: SearchPlan,
  relatedUrls: string[],
  relatedTitles: string[],
  beforeDate?: string | null,
): Promise<SearchHit[]> {
  const hits: SearchHit[] = []
  const fallbackUk = extractKeywordQuery(text, 8)
  const fallbackLatin = extractLatinQuery(text, 6)

  const ukQueries = uniqQueries([...plan.queriesUk, fallbackUk, ...plan.entities], 3)
  const enQueries = uniqQueries(
    [...plan.queriesEn, ...plan.entities.filter((e) => /[A-Za-z]/.test(e)), fallbackLatin],
    2,
  )

  if (relatedUrls.length) {
    hits.push(...relatedUrls.slice(0, 3).map((u, i) => ({ url: u, title: relatedTitles[i] })))
  }

  const toDate =
    beforeDate && /^\d{4}-\d{2}-\d{2}$/.test(beforeDate) ? beforeDate : undefined
  const qUk = ukQueries[0] || ""
  const qEn = enQueries[0] || ""
  const qUkDated = toDate && qUk ? `${qUk} before:${toDate}` : qUk
  const qEnDated = toDate && qEn ? `${qEn} before:${toDate}` : qEn
  const newsParamsUk: Record<string, string> = { q: qUk, language: "uk" }
  if (toDate) newsParamsUk.to = toDate
  const batches = await Promise.all([
    qUkDated
      ? googleNewsRssSearch(qUkDated, { hl: "uk", gl: "UA", ceid: "UA:uk" }, 4)
      : Promise.resolve([]),
    qUk ? newsApiFetch(newsParamsUk, 3) : Promise.resolve([]),
    qEnDated
      ? googleNewsRssSearch(qEnDated, { hl: "en", gl: "US", ceid: "US:en" }, 2)
      : Promise.resolve([]),
  ])

  for (const b of batches) hits.push(...b)
  return dedupeHits(hits, 4)
}

type SourceText = {
  url: string
  title?: string
  text: string
  publisher?: string
  verificationScope: "search_snippet" | "full_page"
}

async function gatherSourceTexts(hits: SearchHit[], maxPages = 3): Promise<SourceText[]> {
  const out: SourceText[] = []
  for (const h of hits) {
    const sn = (h.snippet ?? h.title ?? "").trim()
    if (sn.length < 15) continue
    out.push({
      url: h.url,
      title: h.title,
      publisher: h.publisher,
      text: `SEARCH HIT: ${String(h.title ?? "").slice(0, 200)}\n${sn.slice(0, 700)}`,
      verificationScope: "search_snippet",
    })
    if (out.length >= maxPages) break
  }
  return out
}

function clamp01(n: number) {
  if (!Number.isFinite(n)) return 0
  return Math.max(0, Math.min(1, n))
}

function normalizePercents(
  truth: number,
  fake: number,
): { truthPercent: number; fakePercent: number } | null {
  if (!Number.isFinite(truth) || !Number.isFinite(fake)) return null
  if (truth < 0 || truth > 100 || fake < 0 || fake > 100) return null
  let t = truth
  let f = fake
  const sum = t + f
  if (sum <= 0) return null
  if (Math.abs(sum - 100) > 0.05) {
    t = (t / sum) * 100
    f = 100 - t
  }
  return {
    truthPercent: Math.round(t * 10) / 10,
    fakePercent: Math.round(f * 10) / 10,
  }
}

function normalizeMatchText(s: string): string {
  return String(s || "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim()
}

function quoteMatchesSource(quote: string, sourceText: string): boolean {
  const q = normalizeMatchText(quote)
  const src = normalizeMatchText(sourceText)
  if (q.length < 12 || !src) return false
  return src.includes(q)
}

function verifyCitations(
  citations: { sourceIndex: number; url: string; quote: string }[],
  sourceTexts: SourceText[],
): {
  sourceIndex: number
  url: string
  quote: string
  verificationScope: "search_snippet" | "full_page"
}[] {
  const out: {
    sourceIndex: number
    url: string
    quote: string
    verificationScope: "search_snippet" | "full_page"
  }[] = []
  for (const c of citations) {
    const idx = sourceTexts.findIndex(
      (s) => s.url === c.url || s.url.includes(c.url) || c.url.includes(s.url),
    )
    if (idx < 0) continue
    const src = sourceTexts[idx]
    const blob = `${src.title ?? ""}\n${src.publisher ?? ""}\n${src.text}`
    if (!quoteMatchesSource(c.quote, blob)) continue
    out.push({
      sourceIndex: idx,
      url: src.url,
      quote: c.quote.slice(0, 220),
      verificationScope: src.verificationScope,
    })
  }
  return out
}

function computeSummary(claims: FactCheckClaim[]) {
  const supported = claims.filter((c) => c.verdict === "SUPPORTED").length
  const contradicted = claims.filter((c) => c.verdict === "CONTRADICTED").length
  const notEnoughInfo = claims.filter((c) => c.verdict === "NOT_ENOUGH_INFO").length
  const denom = supported + contradicted + notEnoughInfo
  const evidenceScore = denom ? supported / denom : 0
  return { supported, contradicted, notEnoughInfo, evidenceScore: Number(evidenceScore.toFixed(3)) }
}

function filterInventedOfficeFlags(flags: string[]): string[] {
  
  const bad =
    /(не\s+є|не\s+був|не\s+являється).{0,60}(держсекретар|державн\w+\s+секретар|президент|прем'?єр|secretary of state|president|prime minister)/i
  return flags.filter((f) => !bad.test(f))
}

function dedupeHits(hits: SearchHit[], limit: number): SearchHit[] {
  const seen = new Set<string>()
  const out: SearchHit[] = []
  for (const h of hits) {
    const u = (h.url || "").split("#", 1)[0]
    if (!u || seen.has(u)) continue
    seen.add(u)
    out.push({ ...h, url: u })
    if (out.length >= limit) break
  }
  return out
}

function buildFactCheckPayload(
  llm: LlmConfig,
  parsed: any,
  sourceTexts: SourceText[],
): FactCheckResponse {
  const hasEvidence = sourceTexts.length > 0
  const plausibilityOnly = hasEvidence ? false : Boolean(parsed?.plausibility_only)
  const overallIn = parsed?.overall ?? {}
  const normalized = normalizePercents(
    Number(overallIn?.truth_percent),
    Number(overallIn?.fake_percent),
  )
  let overallVerdict = overallIn?.verdict
  if (!hasEvidence && (overallVerdict === "LIKELY_TRUE" || overallVerdict === "LIKELY_FALSE")) {
    overallVerdict = "NOT_ENOUGH_INFO"
  }
  if (!normalized) {
    overallVerdict = "NOT_ENOUGH_INFO"
  }
  const overall = normalized
    ? {
        truthPercent: normalized.truthPercent,
        fakePercent: normalized.fakePercent,
        confidence: clamp01(Number(overallIn?.confidence ?? 0)),
        verdict: overallVerdict,
        summary: String(overallIn?.summary ?? "").trim(),
      }
    : {
        truthPercent: 50,
        fakePercent: 50,
        confidence: 0,
        verdict: "NOT_ENOUGH_INFO" as const,
        summary: String(overallIn?.summary ?? "").trim() || "Оцінки поза діапазоном 0–100 відхилено.",
      }

  const analysis = String(parsed?.analysis ?? "").trim()
  const redFlags = filterInventedOfficeFlags(
    Array.isArray(parsed?.red_flags)
      ? parsed.red_flags.map((x: any) => String(x ?? "").trim()).filter(Boolean).slice(0, 8)
      : [],
  )

  const claimsIn = Array.isArray(parsed?.key_claims) ? parsed.key_claims : []
  const claims: FactCheckClaim[] = claimsIn
    .map((c: any) => {
      const ev = Array.isArray(c?.evidence) ? c.evidence : []
      const rawCitations = ev
        .map((e: any) => ({
          sourceIndex: -1,
          url: String(e?.url ?? "").trim(),
          quote: String(e?.quote ?? "").trim(),
        }))
        .filter((x: any) => x.url && x.quote)
      const citations = verifyCitations(rawCitations, sourceTexts)
      const confidence = clamp01(Number(c?.confidence ?? 0))
      let verdict = c?.status
      if (
        (verdict === "SUPPORTED" || verdict === "CONTRADICTED") &&
        citations.length === 0
      ) {
        verdict = "NOT_ENOUGH_INFO"
      }
      return {
        claim: String(c?.claim ?? "").trim(),
        verdict,
        confidence,
        citations,
        note: c?.reason ? String(c.reason).trim() : undefined,
      }
    })
    .filter(
      (c: FactCheckClaim) =>
        c.claim &&
        (c.verdict === "SUPPORTED" || c.verdict === "CONTRADICTED" || c.verdict === "NOT_ENOUGH_INFO"),
    )
    .slice(0, 10)

  return {
    provider: llm.label,
    model: llm.model,
    plausibilityOnly,
    overall:
      overall &&
      (overall.verdict === "LIKELY_TRUE" ||
        overall.verdict === "LIKELY_FALSE" ||
        overall.verdict === "MIXED" ||
        overall.verdict === "NOT_ENOUGH_INFO")
        ? overall
        : undefined,
    analysis: analysis || undefined,
    redFlags: redFlags.length ? redFlags : undefined,
    summary: computeSummary(claims),
    claims,
    sourcesUsed: sourceTexts.map((s) => ({ url: s.url, title: s.title })),
  }
}

export async function POST(req: NextRequest) {
  let body: {
    text?: string
    relatedUrls?: string[]
    relatedTitles?: string[]
    verificationMode?: string
    publishedAt?: string | null
    referenceDate?: string | null
    
    skipConsensus?: boolean
    
    providers?: string[]
  }
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: "Некоректний JSON" }, { status: 400 })
  }

  const resolved = resolveLlmConfigs()
  if ("error" in resolved) {
    return NextResponse.json({ error: resolved.error }, { status: 500 })
  }
  const wanted = (body.providers ?? [])
    .map((p) => String(p || "").trim().toLowerCase())
    .filter(Boolean)
  let llms = resolved
  if (wanted.length) {
    const filtered = resolved.filter((c) =>
      wanted.some((w) => c.provider === w || c.label.toLowerCase().includes(w)),
    )
    if (filtered.length) llms = filtered
  }

  const text = (body.text ?? "").trim()
  if (text.length < 40) {
    return NextResponse.json({ error: "Текст занадто короткий для фактчеку" }, { status: 400 })
  }

  const verificationMode =
    (body.verificationMode ?? "").trim().toLowerCase() === "today" ? "today" : "publication"
  const publishedAt = String(body.publishedAt ?? "").trim() || null
  const referenceDate = String(body.referenceDate ?? "").trim() || publishedAt

  const truncatedArticle = text.slice(0, 1_800)

  const latinQ = extractLatinQuery(truncatedArticle, 6)
  const plan: SearchPlan = {
    claims: [],
    queriesUk: uniqQueries([extractKeywordQuery(truncatedArticle, 8)], 2),
    queriesEn: uniqQueries([latinQ], 2),
    entities: latinQ ? latinQ.split(" ").filter(Boolean).slice(0, 6) : [],
  }

  const relatedUrls = Array.isArray(body.relatedUrls) ? body.relatedUrls.filter(Boolean) : []
  const relatedTitles = Array.isArray(body.relatedTitles) ? body.relatedTitles : []
  const evidenceBefore =
    verificationMode === "publication" ? referenceDate || publishedAt : null
  const hits = await collectEvidenceHits(
    text,
    plan,
    relatedUrls,
    relatedTitles,
    evidenceBefore,
  )
  const sourceTexts = await gatherSourceTexts(hits, 3)

  const sourcesBlock = sourceTexts
    .map(
      (s, idx) =>
        `SOURCE[${idx}]\nURL: ${s.url}\nPUBLISHER: ${s.publisher ?? ""}\nTITLE: ${s.title ?? ""}\nTEXT:\n${s.text}\n`,
    )
    .join("\n\n")

  const todayIso = new Date().toISOString().slice(0, 10)
  const todayUk = new Date().toLocaleDateString("uk-UA", {
    year: "numeric",
    month: "long",
    day: "numeric",
  })
  const anchorIso = referenceDate || publishedAt || todayIso
  const anchorUk = (() => {
    const d = new Date(`${anchorIso}T12:00:00`)
    if (Number.isNaN(d.getTime())) return anchorIso
    return d.toLocaleDateString("uk-UA", { year: "numeric", month: "long", day: "numeric" })
  })()
  const timeContextRules =
    verificationMode === "publication"
      ? [
          `РЕЖИМ ЧАСУ: на момент публікації.`,
          `ОПОРНА ДАТА ПЕРЕВІРКИ: ${anchorUk} (${anchorIso}).` +
            (publishedAt ? ` Дата зі сторінки: ${publishedAt}.` : ""),
          `Сьогодні реально: ${todayUk} (${todayIso}) — це ЛИШЕ довідка, НЕ критерій спростування.`,
          "- Оцінюй твердження так, ніби ти перевіряєш новину В ДЕНЬ ПУБЛІКАЦІЇ (опорна дата), а не «з майбутнього».",
          "- ЗАБОРОНЕНО спростовувати майбутні/майбутні-відносно-публікації події лише тому, що «сьогодні вже пізніше».",
          "- Приклад: стаття 16 жовтня пише «матч 17 жовтня» — на момент публікації це НЕ фейк і НЕ CONTRADICTED через те, що зараз 19-е.",
          "- CONTRADICTED по датах лише якщо дата була хибною вже на момент публікації (суперечить джерелам того часу / внутрішньо абсурдна).",
          "- Не плутай «подія вже минула сьогодні» з «новина була неправдивою тоді».",
        ]
      : [
          `РЕЖИМ ЧАСУ: на сьогоднішній день.`,
          `ОПОРНА ДАТА ПЕРЕВІРКИ: ${todayUk} (${todayIso}).`,
          publishedAt ? `Дата публікації матеріалу: ${publishedAt}.` : "",
          "- Оцінюй актуальність формулювань відносно СЬОГОДНІ.",
          "- Застарілі твердження (старі дати подій як майбутні) можуть бути CONTRADICTED, якщо вони вже не відповідають фактам на сьогодні.",
        ]

  const system = [
    "Ти аналітик фейкових новин і фактчекер. Працюєш з БУДЬ-ЯКОЮ темою універсально.",
    "Поверни ЛИШЕ валідний JSON (без markdown).",
    "",
    ...timeContextRules.filter(Boolean),
    "",
    "Критично: твої навчальні дані ЗАСТАРІЛІ щодо політиків, посад, воєн і дат.",
    "- ЗАБОРОНЕНО спростовувати/підтверджувати хто «є / не є» президентом, держсекретарем, прем'єром тощо з пам'яті моделі.",
    "- Посади й імена посадовців: лише SOURCES або BACKGROUND (Вікіпедія). Якщо там немає — NOT_ENOUGH_INFO, НЕ red_flag.",
    "- НЕ пиши в red_flags «X не є держсекретарем/президентом», якщо це не доведено SOURCES.",
    "- Для свіжих подій: якщо джерел мало — NOT_ENOUGH_INFO / MIXED, а не LIKELY_FALSE через невідомість.",
    "",
    "МОВА (обов'язково):",
    "- Усі тексти для користувача — українською: overall.summary, analysis, red_flags, key_claims.claim, key_claims.reason, notes.",
    "- Цитати з джерел (evidence.quote) залишай мовою оригіналу.",
    "- Ключі JSON залишай англійськими, як у схемі.",
    "",
    "Схема:",
    "{",
    '  "plausibility_only": boolean,',
    '  "overall": {',
    '    "truth_percent": number,',
    '    "fake_percent": number,',
    '    "confidence": number,',
    '    "verdict": "LIKELY_TRUE"|"LIKELY_FALSE"|"MIXED"|"NOT_ENOUGH_INFO",',
    '    "summary": string',
    "  },",
    '  "analysis": string,',
    '  "red_flags": [string],',
    '  "key_claims": [',
    "    {",
    '      "claim": string,',
    '      "status": "SUPPORTED"|"CONTRADICTED"|"NOT_ENOUGH_INFO",',
    '      "confidence": number,',
    '      "evidence": [{"url": string, "quote": string}],',
    '      "reason": string',
    "    }",
    "  ],",
    '  "notes": [string]',
    "}",
    "",
    "Джерела (важливо):",
    "- SOURCES — основні докази з новин. BACKGROUND — довідка про осіб/місця з Вікіпедії (не замінює новини, але блокує застарілі «факти» з пам'яті).",
    "- SOURCES можуть бути українські, регіональні, спеціалізовані та міжнародні ЗМІ — усі рівноцінні.",
    "- Відсутність новини в BBC/Reuters/AP НЕ означає фейк.",
    "- НЕ роби головним аргументом «немає в BBC/західних ЗМІ», якщо є інші релевантні SOURCES.",
    "",
    "Що оцінювати:",
    "- Якщо головна подія (хто / що / де / коли) підтверджена незалежними джерелами з SOURCES — став LIKELY_TRUE і truth_percent ≥ 70.",
    "- Непідтверджені другорядні деталі знижують confidence і можуть дати NOT_ENOUGH_INFO по окремих claims, але НЕ перетворюють підтверджену подію на LIKELY_FALSE і не тягни overall.fake_percent вище 40, якщо ядро правдиве.",
    "- Не плутай «новину передрукували» з «новина правдива». Копіювання того самого тексту — НЕ доказ правди.",
    "- Перевір логіку й внутрішні суперечності; відсутність джерел ≠ автоматично фейк.",
    "- Джерела — незалежне підтвердження лише якщо додають різні деталі, а не повторюють той самий сюжет.",
    "- CONTRADICTED лише при сильному спростуванні в SOURCES АБО очевидній абсурдності / порушенні законів природи (з урахуванням РЕЖИМУ ЧАСУ вище).",
    "- Шукай ознаки фейку: клікбейт, анонімні «джерела», перебільшення, емоційна маніпуляція, відсутність фактів, вирваний контекст.",
    "- Якщо зовнішніх джерел мало: plausibility_only=true, знизь confidence, частіше NOT_ENOUGH_INFO або MIXED замість LIKELY_FALSE.",
    "- Якщо в SOURCES є релевантні матеріали: plausibility_only=false.",
    "- Не вигадуй «офіційних спростувань», яких немає в SOURCES.",
    "",
    "Статуси тверджень:",
    "- SUPPORTED: факт незалежно підтверджений як правдивий на ОПОРНУ ДАТУ режиму.",
    "- CONTRADICTED: спростований джерелами або явно внутрішньо абсурдний на ОПОРНУ ДАТУ режиму.",
    "- NOT_ENOUGH_INFO: правдивість не встановлено (брак джерел або невпевненість).",
    "- Не став CONTRADICTED лише через те, що з опорної дати вже минув час до «сьогодні».",
    "",
    "Правила чисел:",
    "- truth_percent і fake_percent лише в діапазоні 0–100; їхня сума = 100. Значення поза 0–100 будуть відхилені.",
    "- confidence від 0 до 1 — чесна оцінка моделі, БЕЗ штучного завищення.",
    "- SUPPORTED / CONTRADICTED лише з evidence: url+quote, які реально є в SOURCES.",
    "- evidence.quote має бути дослівним фрагментом із відповідного джерела в SOURCES (повний збіг, без зміни закінчення).",
    "- Якщо цитати з SOURCES немає — став NOT_ENOUGH_INFO, навіть якщо здається правдоподібним.",
    "- Цитати до 220 символів. Не вигадуй URL і цитати.",
    "- analysis: 2–3 речення українською.",
    "- overall.summary: 1–2 речення українською.",
    "- key_claims: до 4 SUPPORTED і до 3 CONTRADICTED, якщо такі є — БЕЗ дублікатів.",
    "- Не дублюй ту саму думку коротшою й довшою версією.",
    "- Якщо спростованих немає — не вигадуй CONTRADICTED.",
    "- claim — коротке твердження з тексту (1 речення). reason — 1 коротке речення чому так.",
  ].join("\n")

  const user = [
    verificationMode === "publication"
      ? `Режим: на момент публікації. Опорна дата: ${anchorUk} (${anchorIso}). Сьогодні (${todayIso}) не використовуй як спростування майбутніх на той момент подій.`
      : `Режим: на сьогодні. Опорна дата: ${todayUk} (${todayIso}).`,
    "Оціни правдивість новини. Коротко. JSON українською.",
    "",
    "TEXT:",
    "<<<",
    truncatedArticle,
    ">>>",
    "",
    "SOURCES:",
    "<<<",
    sourcesBlock || "Немає зовнішніх джерел. Брак джерел ≠ фейк.",
    ">>>",
  ].join("\n")

  const settled = await Promise.allSettled(
    llms.map(async (llm) => {
      const res = await llmJson(llm, [
        { role: "system", content: system },
        { role: "user", content: user },
      ])
      return buildFactCheckPayload(llm, res.parsed, sourceTexts)
    }),
  )

  const factChecks: FactCheckResponse[] = []
  const errors: string[] = []
  for (let i = 0; i < settled.length; i++) {
    const s = settled[i]
    if (s.status === "fulfilled") factChecks.push(s.value)
    else {
      const msg = s.reason instanceof Error ? s.reason.message : String(s.reason)
      errors.push(`${llms[i].label}: ${msg}`)
    }
  }

  if (!factChecks.length) {
    return NextResponse.json(
      { error: errors.join(" | ") || "Усі моделі фактчеку недоступні" },
      { status: 502 },
    )
  }

  let consensus:
    | { summary: string; analysis: string; provider: string; model: string }
    | undefined
  const prefer =
    factChecks.find((fc) => String(fc.provider || "").toLowerCase().includes("gemini")) ||
    factChecks[0]
  if (prefer?.overall?.summary || prefer?.analysis) {
    const summary = prefer.overall?.summary?.trim() || ""
    let analysis = (prefer.analysis || summary).trim()
    const verdicts = Array.from(
      new Set(
        factChecks
          .map((fc) => fc.overall?.verdict)
          .filter(Boolean)
          .map(String),
      ),
    )
    if (verdicts.length > 1) {
      const parts = factChecks
        .map((fc) => `${fc.provider || "LLM"}: ${fc.overall?.verdict}`)
        .filter(Boolean)
      analysis = `${analysis}\n\nРозбіжності провайдерів: ${parts.join("; ")}.`
    }
    consensus = {
      summary: summary || analysis.slice(0, 220),
      analysis: analysis || summary,
      provider: String(prefer.provider || "Фактчек"),
      model: String(prefer.model || ""),
    }
  }

  const primaryResult = factChecks[0]
  return NextResponse.json({
    ...primaryResult,
    factChecks,
    consensus,
    errors: errors.length ? errors : undefined,
  })
}

