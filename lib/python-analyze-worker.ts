import { spawn, type ChildProcessWithoutNullStreams } from "child_process"
import path from "path"
import { pythonChildEnv } from "@/lib/python-env"

type Pending = {
  resolve: (value: unknown) => void
  reject: (err: Error) => void
  timer: ReturnType<typeof setTimeout>
}

let child: ChildProcessWithoutNullStreams | null = null
let starting: Promise<void> | null = null
let nextId = 1
let buffer = ""
const pending = new Map<number, Pending>()

function pythonCommands(): string[][] {
  if (process.env.PYTHON_PATH) return [[process.env.PYTHON_PATH]]
  if (process.platform === "win32") return [["python"], ["py", "-3"]]
  return [["python3"], ["python"]]
}

function rejectAll(err: Error) {
  for (const [id, p] of pending) {
    clearTimeout(p.timer)
    p.reject(err)
    pending.delete(id)
  }
}

function handleLine(line: string) {
  let msg: any
  try {
    msg = JSON.parse(line)
  } catch {
    return
  }
  if (msg?.ready) return
  const id = Number(msg?.id)
  if (!Number.isFinite(id) || !pending.has(id)) return
  const p = pending.get(id)!
  pending.delete(id)
  clearTimeout(p.timer)
  if (msg?.ok) p.resolve(msg.data)
  else p.reject(new Error(String(msg?.error || "Python worker error")))
}

function attach(proc: ChildProcessWithoutNullStreams) {
  buffer = ""
  proc.stdout.setEncoding("utf8")
  proc.stderr.setEncoding("utf8")
  proc.stdout.on("data", (chunk: string) => {
    buffer += chunk
    let idx: number
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).trim()
      buffer = buffer.slice(idx + 1)
      if (line) handleLine(line)
    }
  })
  proc.stderr.on("data", () => {})
  proc.on("exit", () => {
    child = null
    starting = null
    rejectAll(new Error("Python analyze worker exited"))
  })
}

async function spawnWorker(): Promise<void> {
  const script = path.join(process.cwd(), "python", "analyze_url.py")
  let lastErr: unknown
  for (const parts of pythonCommands()) {
    const cmd = parts[0]
    const prefix = parts.slice(1)
    try {
      const proc = spawn(cmd, [...prefix, script, "--worker"], {
        windowsHide: true,
        stdio: ["pipe", "pipe", "pipe"],
        env: pythonChildEnv(),
      }) as ChildProcessWithoutNullStreams
      await new Promise<void>((resolve, reject) => {
        let settled = false
        const timer = setTimeout(() => {
          if (settled) return
          settled = true
          try {
            proc.kill()
          } catch {}
          reject(new Error("Python worker startup timeout"))
        }, 60_000)
        let localBuf = ""
        const onData = (chunk: string) => {
          localBuf += chunk
          if (!localBuf.includes("\n")) return
          const line = localBuf.split("\n").find((l) => l.trim())
          if (!line) return
          try {
            const msg = JSON.parse(line.trim())
            if (msg?.ready || msg?.ok) {
              if (settled) return
              settled = true
              clearTimeout(timer)
              proc.stdout.off("data", onData)
              resolve()
            }
          } catch {}
        }
        proc.stdout.setEncoding("utf8")
        proc.stdout.on("data", onData)
        proc.on("error", (e) => {
          if (settled) return
          settled = true
          clearTimeout(timer)
          reject(e)
        })
        proc.on("exit", (code) => {
          if (settled) return
          settled = true
          clearTimeout(timer)
          reject(new Error(`Python worker exited during startup (${code})`))
        })
      })
      child = proc
      attach(proc)
      return
    } catch (e) {
      lastErr = e
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("Не вдалося запустити Python worker")
}

async function ensureWorker(): Promise<ChildProcessWithoutNullStreams> {
  if (child && !child.killed) return child
  if (!starting) {
    starting = spawnWorker().finally(() => {
      starting = null
    })
  }
  await starting
  if (!child) throw new Error("Python worker not available")
  return child
}

export async function analyzeViaWorker(
  payload: {
    op: "text" | "url"
    text?: string
    url?: string
    verificationMode?: string
    timeout?: number
  },
  timeoutMs = 120_000,
): Promise<any> {
  const proc = await ensureWorker()
  const id = nextId++
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id)
      reject(new Error("Request Timeout"))
    }, timeoutMs)
    pending.set(id, { resolve, reject, timer })
    try {
      proc.stdin.write(`${JSON.stringify({ id, ...payload })}\n`, "utf8")
    } catch (e) {
      pending.delete(id)
      clearTimeout(timer)
      reject(e instanceof Error ? e : new Error(String(e)))
    }
  })
}
