import { execFile, spawn } from "child_process"
import { promisify } from "util"
import path from "path"
import { NextRequest, NextResponse } from "next/server"

const execFileAsync = promisify(execFile)

function pythonCommands(): string[][] {
  if (process.platform === "win32") {
    return [["python"], ["py", "-3"]]
  }
  return [["python3"], ["python"]]
}

const MODES = ["publication", "today"] as const

function runPythonWithStdin(
  command: string,
  commandPrefix: string[],
  args: string[],
  stdinPayload: string,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const fullArgs = [...commandPrefix, ...args]
    const child = spawn(command, fullArgs, {
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    })
    let stdout = ""
    let stderr = ""
    child.stdout?.on("data", (c: Buffer) => {
      stdout += c.toString()
    })
    child.stderr?.on("data", (c: Buffer) => {
      stderr += c.toString()
    })
    child.on("error", reject)
    child.on("close", (code) => {
      if (code !== 0 && !stdout.trim()) {
        const err = new Error(stderr || `Python exited with code ${code}`) as Error & {
          stdout?: string
        }
        err.stdout = stdout
        reject(err)
        return
      }
      resolve(stdout)
    })
    child.stdin?.write(stdinPayload, "utf8")
    child.stdin?.end()
  })
}

async function runStdinAnalyze(script: string, payload: object, attempts: string[][]): Promise<string> {
  const stdinJson = JSON.stringify(payload)
  if (process.env.PYTHON_PATH) {
    try {
      return await runPythonWithStdin(process.env.PYTHON_PATH, [], [script, "--json", "--stdin"], stdinJson)
    } catch {}
  }
  let lastError: unknown
  for (const parts of attempts) {
    const cmd = parts[0]
    const prefix = parts.slice(1)
    try {
      return await runPythonWithStdin(cmd, prefix, [script, "--json", "--stdin"], stdinJson)
    } catch (e) {
      lastError = e
    }
  }
  throw lastError ?? new Error("Не вдалося запустити Python для аналізу тексту")
}

export async function POST(request: NextRequest) {
  let body: { url?: string; rawText?: string; verificationMode?: string }
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: "Некоректний JSON" }, { status: 400 })
  }

  const url = typeof body.url === "string" ? body.url.trim() : ""
  const rawText = typeof body.rawText === "string" ? body.rawText.trim() : ""

  if (url && rawText) {
    return NextResponse.json({ error: "Передайте або посилання (url), або текст (rawText), не обидва" }, { status: 400 })
  }
  if (!url && !rawText) {
    return NextResponse.json({ error: "Потрібен параметр url або rawText" }, { status: 400 })
  }

  const rawMode = (body.verificationMode ?? "publication").trim().toLowerCase()
  const verificationMode = MODES.includes(rawMode as (typeof MODES)[number]) ? rawMode : "publication"

  const script = path.join(process.cwd(), "python", "analyze_url.py")
  const attempts = pythonCommands()
  const env = { ...process.env }

  try {
    let stdout: string
    if (rawText) {
      stdout = await runStdinAnalyze(script, { text: rawText, verificationMode: "today" }, attempts)
    } else {
      const scriptArgs = [script, url, "--json", "--mode", verificationMode]
      if (process.env.PYTHON_PATH) {
        const { stdout: out } = await execFileAsync(process.env.PYTHON_PATH, scriptArgs, {
          maxBuffer: 50 * 1024 * 1024,
          timeout: 120_000,
          windowsHide: true,
          env,
        })
        stdout = out
      } else {
        let lastError: unknown
        let captured: string | undefined
        for (const parts of attempts) {
          const cmd = parts[0]
          const prefix = parts.slice(1)
          try {
            const { stdout: out } = await execFileAsync(cmd, [...prefix, ...scriptArgs], {
              maxBuffer: 50 * 1024 * 1024,
              timeout: 120_000,
              windowsHide: true,
              env,
            })
            captured = out
            break
          } catch (e: unknown) {
            lastError = e
          }
        }
        if (captured === undefined) throw lastError
        stdout = captured
      }
    }
    return parseStdout(stdout)
  } catch (e: unknown) {
    const fromStdout = tryParseExecStdout(e)
    if (fromStdout) return fromStdout
    const msg = e instanceof Error ? e.message : String(e)
    const hint =
      "Перевірте, що Python у PATH і встановлено залежності: pip install -r requirements.txt"
    return NextResponse.json({ error: `${msg}. ${hint}` }, { status: 500 })
  }
}

function tryParseExecStdout(e: unknown): NextResponse | null {
  const stdout =
    e && typeof e === "object" && "stdout" in e
      ? String((e as { stdout?: Buffer | string }).stdout ?? "")
      : ""
  if (!stdout.trim()) return null
  try {
    const data = JSON.parse(stdout.trim()) as { error?: string }
    if (data.error) {
      return NextResponse.json(data, { status: 502 })
    }
  } catch {
    
  }
  return null
}

function parseStdout(stdout: string): NextResponse {
  const line = stdout.trim()
  if (!line) {
    return NextResponse.json({ error: "Порожня відповідь від Python" }, { status: 502 })
  }
  let data: {
    error?: string
    text?: string
    images?: string[]
    videos?: string[]
    publishedAt?: string | null
    inputKind?: string
    verificationMode?: string
    referenceDate?: string
    modeTitle?: string
    modeDetail?: string
    baseTruthPercentage?: number
    baseFakePercentage?: number
    scoreNote?: string
    truthPercentage?: number
    fakePercentage?: number
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
    trueParts?: string[]
    fakeParts?: string[]
    sources?: string[]
    relatedArticles?: { title: string; url: string; snippet?: string; publisher?: string }[]
    readingSearchQuery?: string
  }
  try {
    data = JSON.parse(line) as typeof data
  } catch {
    return NextResponse.json(
      { error: "Не вдалося розпарсити JSON від Python", raw: line.slice(0, 500) },
      { status: 502 },
    )
  }
  if (data.error) {
    return NextResponse.json(data, { status: 502 })
  }
  return NextResponse.json(data)
}
