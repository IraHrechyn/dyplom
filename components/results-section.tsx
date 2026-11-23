"use client"

import { useState } from "react"
import { PieChart, Pie, Cell } from "recharts"

function formatPct(n: number) {
  return Number.isInteger(n) ? String(n) : n.toFixed(1)
}

type Fragment = {
  text: string
  trustPercent: number
  riskPercent: number
}

type FactClaim = {
  claim: string
  verdict: "SUPPORTED" | "CONTRADICTED" | "NOT_ENOUGH_INFO"
  confidence: number
  note?: string
  citations?: { sourceIndex: number; url: string; quote: string }[]
  provider?: string
}

function ClaimCard({ item, kind }: { item: FactClaim; kind: "true" | "false" }) {
  const shell =
    kind === "true"
      ? "border-emerald-800/50 bg-emerald-950/20"
      : "border-red-800/50 bg-red-950/25"
  const badge = kind === "true" ? "Підтверджено" : "Спростовано"
  const badgeColor = kind === "true" ? "text-emerald-400/90" : "text-red-400"
  const bar = kind === "true" ? "bg-emerald-500" : "bg-red-600"
  const confPct = Math.round(Math.max(0, Math.min(1, item.confidence)) * 100)
  const citations = (item.citations ?? []).filter((c) => c.url && c.quote)

  return (
    <div className={`rounded-xl border p-5 text-left h-full flex flex-col ${shell}`}>
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <span className={`text-xs font-bold uppercase tracking-wide ${badgeColor}`}>{badge}</span>
        <span className="text-neutral-500 text-xs">
          впевненість: <span className="text-white font-semibold">{confPct}%</span>
        </span>
        {item.provider && (
          <span className="text-neutral-600 text-xs">{item.provider}</span>
        )}
      </div>
      <div className="h-1.5 w-full rounded-full bg-black/60 overflow-hidden mb-3">
        <div
          className={`h-full rounded-full ${bar}`}
          style={{ width: `${Math.min(100, Math.max(8, confPct))}%` }}
        />
      </div>
      <p className="text-neutral-100 text-sm leading-relaxed font-medium">{item.claim}</p>
      {citations.length > 0 && (
        <div className="mt-3 space-y-2 border-t border-neutral-800/80 pt-3">
          {citations.slice(0, 2).map((c, i) => (
            <div key={`${c.url}-${i}`} className="text-xs text-neutral-400 leading-relaxed">
              <a
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-red-400/90 hover:text-red-300 break-all"
              >
                {c.url.replace(/^https?:\/\//, "").slice(0, 72)}
              </a>
              <p className="mt-1 text-neutral-500 italic">«{c.quote}»</p>
            </div>
          ))}
        </div>
      )}
      {item.note && (
        <p className="mt-3 text-xs text-neutral-500 leading-relaxed border-t border-neutral-800/80 pt-3">
          {item.note}
        </p>
      )}
    </div>
  )
}

function claimTokens(claim: string): Set<string> {
  return new Set(
    claim
      .toLowerCase()
      .replace(/[«»"'“”]/g, "")
      .split(/[^\p{L}\p{N}]+/u)
      .filter((w) => w.length >= 4),
  )
}

function claimsNearDuplicate(a: string, b: string): boolean {
  const ta = claimTokens(a)
  const tb = claimTokens(b)
  if (!ta.size || !tb.size) return false
  let inter = 0
  for (const t of ta) if (tb.has(t)) inter++
  const union = ta.size + tb.size - inter
  const jaccard = union ? inter / union : 0
  const overlapA = inter / ta.size
  const overlapB = inter / tb.size
  return jaccard >= 0.55 || overlapA >= 0.75 || overlapB >= 0.75
}

function dedupeClaims(items: FactClaim[]): FactClaim[] {
  const sorted = [...items].sort((a, b) => b.confidence - a.confidence)
  const out: FactClaim[] = []
  for (const c of sorted) {
    if (out.some((x) => claimsNearDuplicate(x.claim, c.claim))) continue
    out.push(c)
  }
  return out
}

function mergeFactClaims(
  factChecks: { claims?: FactClaim[]; provider?: string }[],
): {
  supported: FactClaim[]
  contradicted: FactClaim[]
} {
  const supported: FactClaim[] = []
  const contradicted: FactClaim[] = []

  for (const fc of factChecks) {
    for (const c of fc.claims ?? []) {
      if (!c.claim.trim()) continue
      const citations = (c.citations ?? []).filter((x) => x.url && x.quote)
      if (!citations.length) continue
      const item = { ...c, citations, provider: c.provider || fc.provider }
      if (c.verdict === "SUPPORTED") supported.push(item)
      else if (c.verdict === "CONTRADICTED") contradicted.push(item)
    }
  }

  const supportedD = dedupeClaims(supported)
  const contradictedD = dedupeClaims(contradicted)
  const supportedOk = supportedD
    .filter((s) => !contradictedD.some((x) => claimsNearDuplicate(s.claim, x.claim)))
    .slice(0, 6)
  const contradictedOk = contradictedD
    .filter((c) => !supportedD.some((x) => claimsNearDuplicate(c.claim, x.claim)))
    .slice(0, 6)
  return {
    supported: supportedOk,
    contradicted: contradictedOk,
  }
}

interface ResultsSectionProps {
  result: {
    truthPercentage: number
    fakePercentage: number
    trueParts: string[]
    fakeParts: string[]
    sources: string[]
    verificationMode?: string
    referenceDate?: string
    publishedAt?: string | null
    modeTitle?: string
    modeDetail?: string
    baseTruthPercentage?: number
    baseFakePercentage?: number
    scoreNote?: string
    modelBreakdown?: {
      language?: string
      activeModel?: string
      sklearnFakePercentage?: number | null
      sklearnTruthPercentage?: number | null
      kerasFakePercentage?: number | null
      kerasTruthPercentage?: number | null
      sklearnWeight?: number
      kerasWeight?: number
    }
    reliableFragments?: Fragment[]
    questionableFragments?: Fragment[]
    relatedArticles?: { title: string; url: string; snippet?: string; publisher?: string }[]
    readingSearchQuery?: string
    factCheck?: {
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
      summary: { supported: number; contradicted: number; notEnoughInfo: number; evidenceScore: number }
      claims: {
        claim: string
        verdict: "SUPPORTED" | "CONTRADICTED" | "NOT_ENOUGH_INFO"
        confidence: number
        citations: { sourceIndex: number; url: string; quote: string }[]
        note?: string
      }[]
      sourcesUsed: { url: string; title?: string }[]
    }
    factChecks?: {
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
      summary: { supported: number; contradicted: number; notEnoughInfo: number; evidenceScore: number }
      claims: {
        claim: string
        verdict: "SUPPORTED" | "CONTRADICTED" | "NOT_ENOUGH_INFO"
        confidence: number
        citations: { sourceIndex: number; url: string; quote: string }[]
        note?: string
      }[]
      sourcesUsed: { url: string; title?: string }[]
    }[]
    factCheckConsensus?: {
      summary: string
      analysis: string
      provider?: string
      model?: string
    }
    combined?: {
      combinedFakePercent: number
      combinedTruthPercent: number
      note: string
    }
    factCheckError?: string
  }
}

export default function ResultsSection({ result }: ResultsSectionProps) {
  const [chartAnimKey, setChartAnimKey] = useState(0)

  const chartTruth = result.combined?.combinedTruthPercent ?? result.truthPercentage
  const chartFake = result.combined?.combinedFakePercent ?? result.fakePercentage
  const chartNote = result.combined?.note

  const chartData = [
    { name: "True", value: chartTruth, fill: "#e8e8e8" },
    { name: "Fake", value: chartFake, fill: "#dc2626" },
  ]

  const factChecks =
    result.factChecks && result.factChecks.length
      ? result.factChecks
      : result.factCheck
        ? [result.factCheck]
        : []
  const { supported: trueFacts, contradicted: falseFacts } = mergeFactClaims(factChecks)

  return (
    <section id="results" className="w-full min-h-screen px-4 bg-black border-t border-neutral-900 flex flex-col">
      <div className="max-w-6xl mx-auto w-full py-16 flex-1">
        <div className="mb-12">
          <h2 className="text-4xl font-bold text-white mb-2">Результати аналізу</h2>
          <p className="text-neutral-500">Загальна оцінка тексту та ключові факти з фактчеку</p>
        </div>

        {(result.modeTitle ||
          result.referenceDate ||
          (result.publishedAt != null && result.publishedAt !== "")) && (
          <div className="mb-8 rounded-lg border border-red-900/40 bg-red-950/20 px-5 py-4 text-sm text-neutral-300">
            {result.modeTitle && (
              <p className="font-semibold text-white mb-1">Режим: {result.modeTitle}</p>
            )}
            {result.publishedAt != null && result.publishedAt !== "" && (
              <p>
                <span className="text-neutral-500">Дата зі сторінки: </span>
                <span className="text-white">{result.publishedAt}</span>
              </p>
            )}
            {result.referenceDate && (
              <p className="mt-1">
                <span className="text-neutral-500">Опорна дата для звіту: </span>
                <span className="text-white">{result.referenceDate}</span>
              </p>
            )}
          </div>
        )}

        <div className="mb-12 bg-gradient-to-br from-neutral-900 to-black p-8 rounded-lg border border-neutral-900">
          <h3 className="text-xl font-semibold text-white mb-6">Загальний висновок</h3>
          <div className="flex flex-col md:flex-row md:items-center gap-8 md:gap-12">
            <div
              className="flex justify-center shrink-0 cursor-pointer"
              onMouseEnter={() => setChartAnimKey((k) => k + 1)}
            >
              <PieChart width={280} height={280}>
                <Pie
                  key={chartAnimKey}
                  data={chartData}
                  cx={140}
                  cy={140}
                  innerRadius={70}
                  outerRadius={115}
                  dataKey="value"
                  startAngle={90}
                  endAngle={-270}
                  isAnimationActive
                  animationBegin={80}
                  animationDuration={1800}
                  animationEasing="ease-in-out"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} stroke="transparent" />
                  ))}
                </Pie>
              </PieChart>
            </div>

            <div className="flex-1 space-y-5 min-w-0">
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded" style={{ backgroundColor: "#e8e8e8" }} />
                    <span className="text-gray-400">Надійність</span>
                  </div>
                  <span className="text-4xl font-bold text-white">{formatPct(chartTruth)}%</span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-red-600 rounded" />
                    <span className="text-gray-400">Ризик</span>
                  </div>
                  <span className="text-4xl font-bold text-red-600">{formatPct(chartFake)}%</span>
                </div>
              </div>

              {chartNote && (
                <p className="text-sm text-neutral-500 leading-relaxed max-w-xl">{chartNote}</p>
              )}
            </div>
          </div>
        </div>

        {result.factCheckError && factChecks.length === 0 && (
          <div className="mb-12 rounded-lg border border-red-900/60 bg-red-950/30 p-6">
            <h4 className="text-lg font-semibold text-red-400 mb-2">Фактчек недоступний</h4>
            <p className="text-neutral-400 text-sm leading-relaxed whitespace-pre-wrap">
              {result.factCheckError}
            </p>
            <p className="text-neutral-600 text-xs mt-3">
              Блок міркувань ШІ зʼявляється лише після успішного фактчеку. Спробуйте ще раз через хвилину.
            </p>
          </div>
        )}

        {result.factCheckConsensus && (
          <div className="mb-12 bg-gradient-to-br from-neutral-900 to-black p-6 rounded-lg border border-neutral-700">
            <h4 className="text-lg font-semibold text-white mb-2">Висновок фактчеку</h4>
            <div className="rounded-lg border border-neutral-800 bg-black/40 px-4 py-3">
              {(result.factCheckConsensus.provider || result.factCheckConsensus.model) && (
                <p className="text-xs text-neutral-500 mb-2">
                  Джерело висновку:{" "}
                  <span className="text-neutral-300">
                    {[result.factCheckConsensus.provider, result.factCheckConsensus.model]
                      .filter(Boolean)
                      .join(" / ")}
                  </span>
                </p>
              )}
              <p className="text-sm text-neutral-300">
                <span className="text-red-400 font-semibold">Висновок:</span>{" "}
                {result.factCheckConsensus.summary}
              </p>
              {result.factCheckConsensus.analysis &&
                result.factCheckConsensus.analysis !== result.factCheckConsensus.summary && (
                  <p className="text-sm text-neutral-400 mt-3 leading-relaxed whitespace-pre-wrap">
                    {result.factCheckConsensus.analysis}
                  </p>
                )}
            </div>
          </div>
        )}

        {(trueFacts.length > 0 || falseFacts.length > 0) && (
          <div className="mb-12">
            <h3 className="text-2xl font-bold text-white mb-8">Ключові факти</h3>

            <div
              className={`grid gap-8 lg:gap-10 ${
                trueFacts.length > 0 && falseFacts.length > 0 ? "lg:grid-cols-2" : "grid-cols-1"
              }`}
            >
              {trueFacts.length > 0 && (
                <div>
                  <div className="flex items-center gap-3 mb-4">
                    <span className="inline-block w-1 h-6 bg-emerald-500 rounded" />
                    <div>
                      <h4 className="text-lg font-semibold text-white">Правдиві</h4>
                      <p className="text-xs text-neutral-600">Підтверджені джерелами або логікою перевірки</p>
                    </div>
                  </div>
                  <div className="grid sm:grid-cols-2 gap-4">
                    {trueFacts.map((item, idx) => (
                      <ClaimCard key={`t-${idx}`} item={item} kind="true" />
                    ))}
                  </div>
                </div>
              )}

              {falseFacts.length > 0 && (
                <div>
                  <div className="flex items-center gap-3 mb-4">
                    <span className="inline-block w-1 h-6 bg-red-500 rounded" />
                    <div>
                      <h4 className="text-lg font-semibold text-white">Неправдиві</h4>
                      <p className="text-xs text-neutral-600">Спростовані або явно хибні твердження</p>
                    </div>
                  </div>
                  <div className="grid sm:grid-cols-2 gap-4">
                    {falseFacts.map((item, idx) => (
                      <ClaimCard key={`f-${idx}`} item={item} kind="false" />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="bg-gradient-to-br from-slate-900 to-black p-8 rounded-lg border border-emerald-900/40 newspaper-texture">
          <h3 className="text-xl font-semibold text-white mb-2 flex items-center gap-2">
            <span className="w-1 h-6 bg-emerald-500" />
            Де почитати з теми
          </h3>
          <p className="text-neutral-500 text-sm mb-6 max-w-3xl leading-relaxed">
            Підбірка новин і матеріалів з відкритих джерел за ключовими словами з вашого тексту.
            Це орієнтири для порівняння, а не вердикт редактора.
          </p>
          {(result.relatedArticles?.length ?? 0) > 0 ? (
            <div className="grid md:grid-cols-2 gap-4">
              {result.relatedArticles!.map((art, idx) => (
                <a
                  key={`${art.url}-${idx}`}
                  href={art.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group flex flex-col gap-2 p-5 bg-black/50 rounded-xl border border-gray-800 hover:border-emerald-600/50 transition-colors text-left"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 mt-2 rounded-full bg-emerald-500 flex-shrink-0" />
                    <div>
                      <p className="text-white font-medium group-hover:text-emerald-300 transition-colors leading-snug">
                        {art.title}
                      </p>
                      {art.publisher && (
                        <p className="text-xs text-emerald-600/90 mt-1">{art.publisher}</p>
                      )}
                    </div>
                  </div>
                  {art.snippet && (
                    <p className="text-gray-500 text-sm pl-5 leading-relaxed line-clamp-4">{art.snippet}</p>
                  )}
                  <span className="text-xs text-gray-600 pl-5 group-hover:text-gray-400 truncate">
                    {art.url.replace(/^https?:\/\//, "")}
                  </span>
                </a>
              ))}
            </div>
          ) : (
            <div className="rounded-lg border border-gray-800 bg-black/40 p-6 text-gray-400 text-sm">
              <p className="mb-3">
                Зараз не вдалося знайти готові картки
                {result.readingSearchQuery ? (
                  <>
                    {" "}
                    за запитом <span className="text-white">«{result.readingSearchQuery}»</span>
                  </>
                ) : null}
                . Спробуйте пошук нижче або повторіть аналіз.
              </p>
              {result.readingSearchQuery && (
                <p>
                  Можна шукати самостійно:{" "}
                  <a
                    className="text-emerald-400 hover:underline"
                    href={`https://uk.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(result.readingSearchQuery)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Вікіпедія (UK)
                  </a>
                  {" · "}
                  <a
                    className="text-emerald-400 hover:underline"
                    href={`https://en.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(result.readingSearchQuery)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Wikipedia (EN)
                  </a>
                </p>
              )}
            </div>
          )}
          {(result.sources?.length ?? 0) > 0 && (
            <div className="mt-6 pt-6 border-t border-gray-800">
              <p className="text-xs text-gray-600 uppercase tracking-wide mb-2">Додатково</p>
              <ul className="space-y-1">
                {result.sources.map((s, i) => (
                  <li key={i} className="text-gray-500 text-sm">
                    {s}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
