import { spawn } from "child_process"
import path from "path"
import { NextRequest, NextResponse } from "next/server"
import { pythonChildEnv } from "@/lib/python-env"

export const runtime = "nodejs"

const MAX_BYTES = 12 * 1024 * 1024
const ALLOWED = new Set(["image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif", "image/bmp"])

function pythonCommands(): string[][] {
  if (process.platform === "win32") return [["python"], ["py", "-3"]]
  return [["python3"], ["python"]]
}

function runOcrStdinB64(command: string, prefix: string[], script: string, b64: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, [...prefix, script, "--json", "--stdin-b64"], {
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
      env: pythonChildEnv(),
    })
    let stdout = ""
    let stderr = ""
    child.stdout?.on("data", (c: Buffer) => {
      stdout += c.toString("utf8")
    })
    child.stderr?.on("data", (c: Buffer) => {
      stderr += c.toString("utf8")
    })
    child.on("error", reject)
    child.on("close", (code) => {
      if (code !== 0 && !stdout.trim()) {
        reject(new Error(stderr || `OCR exited with code ${code}`))
        return
      }
      resolve(stdout)
    })
    child.stdin?.write(b64, "utf8")
    child.stdin?.end()
  })
}

async function runOcr(b64: string): Promise<string> {
  const script = path.join(process.cwd(), "python", "ocr_screenshot.py")
  const attempts = process.env.PYTHON_PATH
    ? [[process.env.PYTHON_PATH]]
    : pythonCommands()

  let lastError: unknown
  for (const parts of attempts) {
    const cmd = parts[0]
    const prefix = parts.slice(1)
    try {
      return await runOcrStdinB64(cmd, prefix, script, b64)
    } catch (e) {
      lastError = e
    }
  }
  throw lastError ?? new Error("Не вдалося запустити Python для OCR")
}

export async function POST(req: NextRequest) {
  let form: FormData
  try {
    form = await req.formData()
  } catch {
    return NextResponse.json({ error: "Очікується multipart/form-data зі скріншотом" }, { status: 400 })
  }

  const file = form.get("image")
  if (!file || typeof file === "string") {
    return NextResponse.json({ error: "Додайте файл image (скріншот новини)" }, { status: 400 })
  }

  const mime = (file.type || "").toLowerCase()
  if (mime && !ALLOWED.has(mime) && !mime.startsWith("image/")) {
    return NextResponse.json({ error: "Підтримуються лише зображення (PNG, JPG, WEBP…)" }, { status: 400 })
  }

  const buf = Buffer.from(await file.arrayBuffer())
  if (!buf.length) {
    return NextResponse.json({ error: "Порожній файл" }, { status: 400 })
  }
  if (buf.length > MAX_BYTES) {
    return NextResponse.json({ error: "Файл занадто великий (макс. ~12 МБ)" }, { status: 400 })
  }

  try {
    const b64 = buf.toString("base64")
    const stdout = await runOcr(b64)
    const data = JSON.parse(stdout.trim()) as { text?: string; error?: string }
    if (data.error) {
      return NextResponse.json({ error: data.error }, { status: 422 })
    }
    const text = (data.text || "").trim()
    if (text.length < 15) {
      return NextResponse.json(
        { error: "На скріні замало тексту. Спробуйте чіткіший скріншот або вставте текст вручну." },
        { status: 422 },
      )
    }
    return NextResponse.json({ text, source: "screenshot" })
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    return NextResponse.json({ error: msg || "OCR не вдався" }, { status: 502 })
  }
}
