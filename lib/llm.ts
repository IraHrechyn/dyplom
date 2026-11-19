

export type LlmProvider = "groq" | "openai" | "openrouter" | "anthropic" | "gemini"

export type LlmMessage = { role: "system" | "user" | "assistant"; content: string }

export type LlmConfig = {
  provider: LlmProvider
  model: string
  apiKey: string
  label: string
}

const DEFAULTS: Record<LlmProvider, { model: string; baseUrl?: string }> = {
  groq: { model: "llama-3.3-70b-versatile", baseUrl: "https://api.groq.com/openai/v1" },
  openai: { model: "gpt-4o-mini", baseUrl: "https://api.openai.com/v1" },
  openrouter: { model: "openai/gpt-4o-mini", baseUrl: "https://openrouter.ai/api/v1" },
  anthropic: { model: "claude-sonnet-4-5" },
  gemini: { model: "gemini-flash-lite-latest" },
}

const LABELS: Record<LlmProvider, string> = {
  groq: "Groq",
  openai: "OpenAI",
  openrouter: "OpenRouter",
  anthropic: "Claude",
  gemini: "Gemini",
}

function isProvider(v: string): v is LlmProvider {
  return v === "openai" || v === "openrouter" || v === "anthropic" || v === "groq" || v === "gemini"
}

function keyFor(provider: LlmProvider): string {
  const map: Record<LlmProvider, string> = {
    groq: (process.env.GROQ_API_KEY ?? "").trim(),
    openai: (process.env.OPENAI_API_KEY ?? "").trim(),
    openrouter: (process.env.OPENROUTER_API_KEY ?? "").trim(),
    anthropic: (process.env.ANTHROPIC_API_KEY ?? "").trim(),
    gemini: (process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY ?? "").trim(),
  }
  return map[provider]
}

function modelFor(provider: LlmProvider, singleModel?: string): string {
  const perProvider =
    process.env[`FACTCHECK_MODEL_${provider.toUpperCase()}`]?.trim() ||
    (provider === "groq" ? process.env.GROQ_FACTCHECK_MODEL?.trim() : undefined) ||
    (provider === "gemini" ? process.env.GEMINI_FACTCHECK_MODEL?.trim() : undefined)
  return perProvider || singleModel || DEFAULTS[provider].model
}

function envName(provider: LlmProvider): string {
  if (provider === "groq") return "GROQ_API_KEY"
  if (provider === "openai") return "OPENAI_API_KEY"
  if (provider === "openrouter") return "OPENROUTER_API_KEY"
  if (provider === "gemini") return "GEMINI_API_KEY"
  return "ANTHROPIC_API_KEY"
}

function pickProviderList(): LlmProvider[] {
  const multi = (process.env.FACTCHECK_PROVIDERS ?? "").trim()
  const raw = multi
    ? multi.split(",")
    : [(process.env.FACTCHECK_PROVIDER ?? "groq").trim()]
  const out: LlmProvider[] = []
  for (const part of raw) {
    const p = part.trim().toLowerCase()
    if (isProvider(p) && !out.includes(p)) out.push(p)
  }
  return out.length ? out : ["groq"]
}

export function resolveLlmConfig(): LlmConfig | { error: string } {
  const configs = resolveLlmConfigs()
  if ("error" in configs) return configs
  return configs[0]
}


export function resolveLlmConfigs(): LlmConfig[] | { error: string } {
  const providers = pickProviderList()
  const singleModel = process.env.FACTCHECK_MODEL?.trim()
  const configs: LlmConfig[] = []
  const missing: string[] = []

  for (const provider of providers) {
    const apiKey = keyFor(provider)
    if (!apiKey) {
      missing.push(`${provider} (${envName(provider)})`)
      continue
    }
    configs.push({
      provider,
      model: modelFor(provider, providers.length === 1 ? singleModel : undefined),
      apiKey,
      label: LABELS[provider],
    })
  }

  if (!configs.length) {
    return {
      error: `Немає доступних LLM для фактчеку. Перевірте ключі: ${missing.join(", ") || "не задано"}.`,
    }
  }
  return configs
}


export function resolveProviderConfig(provider: LlmProvider): LlmConfig | { error: string } {
  const apiKey = keyFor(provider)
  if (!apiKey) {
    return { error: `Не задано ${envName(provider)} для «${provider}».` }
  }
  return {
    provider,
    model: modelFor(provider),
    apiKey,
    label: LABELS[provider],
  }
}

function tryParseJsonFromText(raw: string): any | null {
  const s = (raw ?? "").trim()
  if (!s) return null
  try {
    return JSON.parse(s)
  } catch {
    const startObj = s.indexOf("{")
    const startArr = s.indexOf("[")
    const start =
      startObj === -1 ? startArr : startArr === -1 ? startObj : Math.min(startObj, startArr)
    if (start === -1) return null
    const tail = s.slice(start)
    const endObj = tail.lastIndexOf("}")
    const endArr = tail.lastIndexOf("]")
    const end = Math.max(endObj, endArr)
    if (end <= 0) return null
    try {
      return JSON.parse(tail.slice(0, end + 1))
    } catch {
      return null
    }
  }
}

async function openaiCompatibleJson(
  baseUrl: string,
  apiKey: string,
  model: string,
  messages: LlmMessage[],
  extraHeaders?: Record<string, string>,
): Promise<{ parsed: any; raw: string }> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      ...extraHeaders,
    },
    body: JSON.stringify({
      model,
      messages,
      temperature: 0.1,
      max_tokens: 2048,
      response_format: { type: "json_object" },
    }),
    signal: AbortSignal.timeout(45_000),
  })

  const data = (await response.json()) as any
  if (!response.ok) {
    const err = data?.error ?? data
    const msg =
      typeof err === "string"
        ? err
        : typeof err?.message === "string"
          ? err.message
          : JSON.stringify(err)
    throw new Error(msg || `LLM error ${response.status}`)
  }

  const raw = String(data.choices?.[0]?.message?.content ?? "")
  const parsed = tryParseJsonFromText(raw)
  if (!parsed) throw new Error("LLM returned non-JSON")
  return { parsed, raw }
}

async function anthropicJson(
  apiKey: string,
  model: string,
  messages: LlmMessage[],
): Promise<{ parsed: any; raw: string }> {
  const system = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n")
  const userMessages = messages
    .filter((m) => m.role !== "system")
    .map((m) => ({ role: m.role === "assistant" ? "assistant" : "user", content: m.content }))

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      max_tokens: 4096,
      temperature: 0.1,
      system: system || undefined,
      messages: userMessages.length ? userMessages : [{ role: "user", content: "Return JSON." }],
    }),
  })

  const data = (await response.json()) as any
  if (!response.ok) {
    const err = data?.error ?? data
    const msg =
      typeof err === "string"
        ? err
        : typeof err?.message === "string"
          ? err.message
          : JSON.stringify(err)
    throw new Error(msg || `Anthropic error ${response.status}`)
  }

  const raw = String(
    (Array.isArray(data?.content) ? data.content : [])
      .filter((b: any) => b?.type === "text")
      .map((b: any) => b.text)
      .join("\n") ?? "",
  )
  const parsed = tryParseJsonFromText(raw)
  if (!parsed) throw new Error("LLM returned non-JSON")
  return { parsed, raw }
}

async function geminiJsonOnce(
  apiKey: string,
  model: string,
  system: string,
  contents: { role: string; parts: { text: string }[] }[],
): Promise<{ parsed: any; raw: string }> {
  const modelId = model.replace(/^models\//, "")
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
    modelId,
  )}:generateContent?key=${encodeURIComponent(apiKey)}`

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      systemInstruction: system ? { parts: [{ text: system }] } : undefined,
      contents,
      generationConfig: {
        temperature: 0.1,
        maxOutputTokens: 2048,
        responseMimeType: "application/json",
      },
    }),
    signal: AbortSignal.timeout(55_000),
  })

  const data = (await response.json()) as any
  if (!response.ok) {
    const err = data?.error ?? data
    const msg =
      typeof err === "string"
        ? err
        : typeof err?.message === "string"
          ? err.message
          : JSON.stringify(err)
    const e = new Error(msg || `Gemini error ${response.status}`) as Error & { status?: number }
    e.status = response.status
    throw e
  }

  const parts = data?.candidates?.[0]?.content?.parts
  const raw = Array.isArray(parts)
    ? parts.map((p: any) => String(p?.text ?? "")).join("\n")
    : ""
  const parsed = tryParseJsonFromText(raw)
  if (!parsed) throw new Error("LLM returned non-JSON")
  return { parsed, raw }
}

async function geminiJson(
  apiKey: string,
  model: string,
  messages: LlmMessage[],
): Promise<{ parsed: any; raw: string }> {
  const system = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n")
  const contents = messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }))

  if (!contents.length) {
    contents.push({ role: "user", parts: [{ text: "Return JSON." }] })
  }

  const fallbacks = [
    model,
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-3.5-flash",
  ]
  const tried = new Set<string>()
  let lastErr: unknown
  for (const m of fallbacks) {
    const id = m.replace(/^models\//, "")
    if (!id || tried.has(id)) continue
    tried.add(id)
    try {
      return await geminiJsonOnce(apiKey, id, system, contents)
    } catch (e) {
      lastErr = e
      const status = (e as { status?: number })?.status
      const msg = e instanceof Error ? e.message : String(e)
      
      if (status === 503 || status === 404 || /high demand|UNAVAILABLE|NOT_FOUND|no longer available/i.test(msg)) {
        continue
      }
      throw e
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr))
}

export async function llmJson(
  config: LlmConfig,
  messages: LlmMessage[],
): Promise<{ parsed: any; raw: string }> {
  if (config.provider === "anthropic") {
    return anthropicJson(config.apiKey, config.model, messages)
  }
  if (config.provider === "gemini") {
    return geminiJson(config.apiKey, config.model, messages)
  }

  const baseUrl = DEFAULTS[config.provider].baseUrl!
  const extra =
    config.provider === "openrouter"
      ? {
          "HTTP-Referer": "http://localhost:3000",
          "X-Title": "Dyplom Factcheck",
        }
      : undefined

  return openaiCompatibleJson(baseUrl, config.apiKey, config.model, messages, extra)
}
