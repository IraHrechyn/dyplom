"use client"

import { useEffect, useState } from "react"
import HeroSection from "@/components/hero-section"
import InputBlock from "@/components/input-block"
import ResultsSection from "@/components/results-section"

function isHttpUrl(s: string): boolean {
  return /^https?:\/\/\S+$/.test(s.trim())
}

export default function Home() {
  const [analysisResult, setAnalysisResult] = useState<{
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
    reliableFragments?: { text: string; trustPercent: number; riskPercent: number }[]
    questionableFragments?: { text: string; trustPercent: number; riskPercent: number }[]
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
  } | null>(null)

  const [isLoading, setIsLoading] = useState(false)
  const [inputUrl, setInputUrl] = useState("")
  const [screenshotFile, setScreenshotFile] = useState<File | null>(null)
  const [verificationMode, setVerificationMode] = useState<"publication" | "today">("publication")
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)

  useEffect(() => {
    if ("scrollRestoration" in history) {
      history.scrollRestoration = "manual"
    }
    if (window.location.hash) {
      history.replaceState(null, "", window.location.pathname + window.location.search)
    }
    window.scrollTo(0, 0)
  }, [])

  const formatError = (x: unknown): string => {
    if (x == null) return ""
    if (typeof x === "string") return x
    if (x instanceof Error) return x.message
    try {
      return JSON.stringify(x)
    } catch {
      return String(x)
    }
  }

  const handleAnalyze = async () => {
    const trimmed = inputUrl.trim()
    if (!trimmed && !screenshotFile) return

    setAnalysisResult(null)
    setAnalyzeError(null)
    setIsLoading(true)
    try {
      let textForFactcheck = trimmed
      let extractBody: { url?: string; rawText?: string; verificationMode: string }

      if (screenshotFile) {
        const form = new FormData()
        form.append("image", screenshotFile)
        const ocrRes = await fetch("/api/ocr", { method: "POST", body: form })
        const ocrPayload = (await ocrRes.json()) as { text?: string; error?: string }
        if (!ocrRes.ok || ocrPayload.error || !ocrPayload.text) {
          setAnalyzeError(formatError(ocrPayload.error ?? ocrRes.statusText) || "Не вдалося зчитати скрін")
          return
        }
        textForFactcheck = ocrPayload.text
        extractBody = { rawText: ocrPayload.text, verificationMode: "today" }
      } else {
        const isLink = isHttpUrl(trimmed)
        extractBody = isLink
          ? { url: trimmed, verificationMode }
          : { rawText: trimmed, verificationMode: "today" }
      }

      type FactCheckItem = {
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

      const runFactcheck = async (
        text: string,
        opts?: {
          verificationMode?: string
          publishedAt?: string | null
          referenceDate?: string | null
        },
      ) => {
        const fcRes = await fetch("/api/factcheck", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            verificationMode: opts?.verificationMode ?? verificationMode,
            publishedAt: opts?.publishedAt ?? null,
            referenceDate: opts?.referenceDate ?? null,
            skipConsensus: true,
            providers: ["openrouter", "gemini"],
          }),
        })
        const fcRaw = await fcRes.text()
        let fcPayload: any = null
        try {
          fcPayload = fcRaw ? JSON.parse(fcRaw) : null
        } catch {
          return {
            factCheck: undefined as FactCheckItem | undefined,
            factChecks: [] as FactCheckItem[],
            factCheckConsensus: undefined as
              | { summary: string; analysis: string; provider?: string; model?: string }
              | undefined,
            factCheckError: `Фактчек повернув не JSON (HTTP ${fcRes.status})`,
          }
        }
        if (!fcRes.ok || fcPayload?.error) {
          return {
            factCheck: undefined as FactCheckItem | undefined,
            factChecks: [] as FactCheckItem[],
            factCheckConsensus: undefined as
              | { summary: string; analysis: string; provider?: string; model?: string }
              | undefined,
            factCheckError: formatError(fcPayload?.error ?? fcPayload ?? fcRes.statusText),
          }
        }
        const list: FactCheckItem[] =
          Array.isArray(fcPayload.factChecks) && fcPayload.factChecks.length
            ? fcPayload.factChecks
            : [fcPayload]
        return {
          factCheck: list[0],
          factChecks: list,
          factCheckConsensus:
            fcPayload.consensus?.summary || fcPayload.consensus?.analysis
              ? {
                  summary: String(fcPayload.consensus.summary ?? "").trim(),
                  analysis: String(fcPayload.consensus.analysis ?? "").trim(),
                  provider: fcPayload.consensus.provider,
                  model: fcPayload.consensus.model,
                }
              : undefined,
          factCheckError: Array.isArray(fcPayload.errors) && fcPayload.errors.length
            ? fcPayload.errors.join(" | ")
            : undefined,
        }
      }

      const canParallel = Boolean(extractBody.rawText)
      let res: Response
      let payload: {
        error?: string
        text?: string
        images?: string[]
        videos?: string[]
        publishedAt?: string | null
        verificationMode?: string
        referenceDate?: string
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
        reliableFragments?: { text: string; trustPercent: number; riskPercent: number }[]
        questionableFragments?: { text: string; trustPercent: number; riskPercent: number }[]
        relatedArticles?: { title: string; url: string; snippet?: string; publisher?: string }[]
        readingSearchQuery?: string
        truthPercentage?: number
        fakePercentage?: number
        trueParts?: string[]
        fakeParts?: string[]
        sources?: string[]
      }
      let factCheck: FactCheckItem | undefined
      let factChecks: FactCheckItem[] = []
      let factCheckConsensus:
        | { summary: string; analysis: string; provider?: string; model?: string }
        | undefined
      let factCheckError: string | undefined

      if (canParallel && extractBody.rawText) {
        const [extractRes, fcOut] = await Promise.all([
          fetch("/api/extract", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(extractBody),
          }),
          runFactcheck(extractBody.rawText, {
            verificationMode: "today",
            publishedAt: null,
            referenceDate: null,
          }).catch((e) => ({
            factCheck: undefined,
            factChecks: [] as FactCheckItem[],
            factCheckConsensus: undefined,
            factCheckError: formatError(e),
          })),
        ])
        res = extractRes
        {
          const raw = await res.text()
          try {
            payload = (raw ? JSON.parse(raw) : {}) as typeof payload
          } catch {
            setAnalyzeError(`Аналіз повернув не JSON (HTTP ${res.status})`)
            return
          }
        }
        factCheck = fcOut.factCheck
        factChecks = fcOut.factChecks
        factCheckConsensus = fcOut.factCheckConsensus
        factCheckError = fcOut.factCheckError
      } else {
        res = await fetch("/api/extract", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(extractBody),
        })
        {
          const raw = await res.text()
          try {
            payload = (raw ? JSON.parse(raw) : {}) as typeof payload
          } catch {
            setAnalyzeError(`Аналіз повернув не JSON (HTTP ${res.status})`)
            return
          }
        }
      }

      console.log("[аналіз статті]", payload)

      if (!res.ok || payload.error) {
        setAnalyzeError(formatError(payload.error ?? res.statusText) || "Помилка аналізу")
        console.error("[extract]", payload.error ?? res.statusText)
        return
      }

      const truth = payload.truthPercentage ?? 0
      const fake = payload.fakePercentage ?? 0
      textForFactcheck = payload.text ?? textForFactcheck

      if (!canParallel) {
        try {
          const fcOut = await runFactcheck(textForFactcheck, {
            verificationMode: payload.verificationMode ?? verificationMode,
            publishedAt: payload.publishedAt ?? null,
            referenceDate: payload.referenceDate ?? null,
          })
          factCheck = fcOut.factCheck
          factChecks = fcOut.factChecks
          factCheckConsensus = fcOut.factCheckConsensus
          factCheckError = fcOut.factCheckError
        } catch (e) {
          factCheckError = formatError(e)
          console.warn("[factcheck] failed", e)
        }
      } else if (factCheckError) {
        console.warn("[factcheck]", factCheckError)
      }

      const TRUST = {
        openrouter: 0.2,
        gemini: 0.2,
        style: 0.6,
      } as const

      const usable = factChecks.filter((fc) => {
        const o = fc?.overall
        const hasSources = (fc?.sourcesUsed?.length ?? 0) > 0
        if (!o || typeof o.fakePercent !== "number") return false
        if (fc?.plausibilityOnly && !hasSources && o.verdict === "NOT_ENOUGH_INFO") return false
        return true
      })

      const ensembleFake = fake
      const votes: { name: string; fake: number; weight: number }[] = []
      const mb = payload.modelBreakdown
      const styleWeight = TRUST.style
      if (mb?.activeModel === "tfidf" || (mb && Number(mb.sklearnWeight) > Number(mb.kerasWeight))) {
        votes.push({ name: "TF-IDF (UA)", fake: ensembleFake, weight: styleWeight })
      } else if (mb?.activeModel === "keras" || mb) {
        votes.push({ name: "Keras (EN)", fake: ensembleFake, weight: styleWeight })
      } else {
        votes.push({ name: "стиль", fake: ensembleFake, weight: styleWeight })
      }

      for (const fc of usable) {
        const p = (fc.provider || "").toLowerCase()
        let weight = p.includes("gemini")
          ? TRUST.gemini
          : p.includes("openrouter")
            ? TRUST.openrouter
            : 0.35
        if (fc.overall?.verdict === "NOT_ENOUGH_INFO") weight *= 0.7
        votes.push({
          name: fc.provider || "LLM",
          fake: Number(fc.overall!.fakePercent),
          weight,
        })
      }

      let combinedFake = ensembleFake
      let combinedNote = factCheckError
        ? "Фінальна оцінка за стилем тексту. Фактчек зараз недоступний."
        : "Фінальна оцінка за стилем тексту."

      if (votes.length) {
        const sumW = votes.reduce((s, v) => s + v.weight, 0) || 1
        combinedFake =
          Math.round((votes.reduce((s, v) => s + v.fake * v.weight, 0) / sumW) * 10) / 10
        combinedNote = usable.length
          ? "Фінальна оцінка за результатами повної перевірки."
          : factChecks.length
            ? "Фактчек виконано; на діаграмі зараз переважно стильова оцінка."
            : combinedNote
      }

      const combinedTruth = Math.round((100 - combinedFake) * 10) / 10

      const screenshotNote = screenshotFile
        ? `Зчитано зі скріншота «${screenshotFile.name}».`
        : null

      setAnalysisResult({
        truthPercentage: combinedTruth,
        fakePercentage: combinedFake,
        trueParts: [
          ...(screenshotNote ? [screenshotNote] : []),
          ...(payload.trueParts?.length ? payload.trueParts : ["Немає пояснень від моделі."]),
        ],
        fakeParts:
          payload.fakeParts?.length ? payload.fakeParts : ["Немає пояснень від моделі."],
        sources: payload.sources?.length ? payload.sources : [],
        verificationMode: payload.verificationMode,
        referenceDate: payload.referenceDate,
        publishedAt: payload.publishedAt ?? null,
        modeTitle: payload.modeTitle,
        modeDetail: payload.modeDetail,
        baseTruthPercentage: payload.baseTruthPercentage ?? truth,
        baseFakePercentage: payload.baseFakePercentage ?? fake,
        scoreNote: payload.scoreNote,
        reliableFragments: payload.reliableFragments,
        questionableFragments: payload.questionableFragments,
        relatedArticles: payload.relatedArticles,
        readingSearchQuery: payload.readingSearchQuery,
        factCheck,
        factChecks: factChecks.length ? factChecks : undefined,
        factCheckConsensus,
        combined: {
          combinedFakePercent: combinedFake,
          combinedTruthPercent: combinedTruth,
          note: combinedNote,
        },
        factCheckError,
      })
    } catch (e) {
      setAnalyzeError(formatError(e) || "Помилка запиту")
      console.error("[extract] fetch failed", e)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (!analysisResult || isLoading) return
    const t = window.setTimeout(() => {
      document.getElementById("results")?.scrollIntoView({ behavior: "smooth", block: "start" })
    }, 100)
    return () => window.clearTimeout(t)
  }, [analysisResult, isLoading])

  return (
    <main className="min-h-screen bg-background text-foreground newspaper-texture">
      <HeroSection />
      <InputBlock
        onAnalyze={handleAnalyze}
        isLoading={isLoading}
        inputUrl={inputUrl}
        setInputUrl={(v) => {
          setInputUrl(v)
          if (v.trim()) setScreenshotFile(null)
        }}
        verificationMode={verificationMode}
        setVerificationMode={setVerificationMode}
        screenshotFile={screenshotFile}
        setScreenshotFile={(f) => {
          setScreenshotFile(f)
          if (f) setAnalyzeError(null)
        }}
      />
      {analyzeError && (
        <div className="max-w-3xl mx-auto px-4 -mt-8 mb-8">
          <p className="rounded-lg border border-red-900/50 bg-red-950/40 px-4 py-3 text-sm text-red-200">
            {analyzeError}
          </p>
        </div>
      )}
      {analysisResult && <ResultsSection result={analysisResult} />}
    </main>
  )
}
