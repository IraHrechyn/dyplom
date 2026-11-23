"use client"

import { useEffect, useState } from "react"

function isUrl(input: string): boolean {
  return /^https?:\/\/\S+$/.test(input)
}

function imageFileFromClipboard(e: React.ClipboardEvent): File | null {
  const items = e.clipboardData?.items
  if (!items) return null
  for (const item of Array.from(items)) {
    if (item.type.startsWith("image/")) {
      const blob = item.getAsFile()
      if (!blob) continue
      const ext = item.type.split("/")[1] || "png"
      return new File([blob], `screenshot-${Date.now()}.${ext}`, { type: item.type })
    }
  }
  return null
}

function imageFileFromDrop(e: React.DragEvent): File | null {
  const f = e.dataTransfer?.files?.[0]
  if (!f || !f.type.startsWith("image/")) return null
  return f
}

interface InputBlockProps {
  onAnalyze: () => void
  isLoading: boolean
  inputUrl: string
  setInputUrl: (value: string) => void
  verificationMode: "publication" | "today"
  setVerificationMode: (mode: "publication" | "today") => void
  screenshotFile: File | null
  setScreenshotFile: (file: File | null) => void
}

export default function InputBlock({
  onAnalyze,
  isLoading,
  inputUrl,
  setInputUrl,
  verificationMode,
  setVerificationMode,
  screenshotFile,
  setScreenshotFile,
}: InputBlockProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const showModeToggles = !screenshotFile && isUrl(inputUrl.trim())

  useEffect(() => {
    if (!screenshotFile) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(screenshotFile)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [screenshotFile])

  const acceptScreenshot = (file: File) => {
    setScreenshotFile(file)
    setInputUrl("")
  }

  return (
    <section
      id="analyze"
      className="relative w-full min-h-screen px-4 bg-black border-t border-neutral-900 flex items-center overflow-hidden"
    >
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
        <div className="analyze-glow analyze-glow-a absolute -top-24 left-1/4 h-72 w-72 rounded-full bg-red-600/20 blur-3xl" />
        <div className="analyze-glow analyze-glow-b absolute bottom-10 right-1/5 h-80 w-80 rounded-full bg-red-500/15 blur-3xl" />
        <div className="analyze-glow analyze-glow-c absolute top-1/2 left-1/2 h-64 w-96 -translate-x-1/2 -translate-y-1/2 rounded-full bg-red-700/10 blur-3xl" />
        <div className="analyze-glow analyze-glow-left absolute top-1/3 -left-8 h-40 w-28 rounded-full bg-red-600/25 blur-2xl" />
        <div className="analyze-glow analyze-glow-right absolute top-1/2 -right-8 h-44 w-28 rounded-full bg-red-500/25 blur-2xl" />
        <div className="analyze-glow analyze-glow-left-b absolute bottom-1/4 left-2 h-28 w-20 rounded-full bg-red-700/20 blur-2xl" />
        <div className="analyze-glow analyze-glow-right-b absolute top-1/4 right-4 h-24 w-16 rounded-full bg-red-600/20 blur-2xl" />
      </div>

      <div className="relative z-10 max-w-3xl mx-auto w-full py-16">
        <div className="mb-8 text-center">
          <h2 className="text-4xl font-bold text-white mb-2">Посилання, текст або скрін</h2>
          <p className="text-neutral-500">
            Вставте URL, текст або скріншот (Ctrl+V) — система зчитає зміст і оцінить ризик фейку
          </p>
        </div>

        <div
          className={`relative bg-black rounded-lg p-1 min-h-[60px] ${
            dragOver ? "ring-2 ring-red-500/70" : ""
          }`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            const file = imageFileFromDrop(e)
            if (file) acceptScreenshot(file)
          }}
        >
          {!screenshotFile ? (
            <textarea
              id="analyze-input"
              placeholder="https://… · текст статті · або Ctrl+V зі скріном"
              value={inputUrl}
              onChange={(e) => setInputUrl(e.target.value)}
              rows={3}
              className="w-full min-h-[72px] resize-y bg-neutral-950 text-white px-5 py-3 rounded-lg border border-neutral-800 focus:outline-none focus:border-red-500 placeholder-neutral-600 text-base caret-red-500"
              onPaste={(e) => {
                const file = imageFileFromClipboard(e)
                if (!file) return
                e.preventDefault()
                acceptScreenshot(file)
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) onAnalyze()
              }}
            />
          ) : (
            <div className="relative flex items-center gap-3 w-full min-h-[72px] px-3 rounded-lg border border-neutral-800 bg-neutral-950">
              {previewUrl && (
                
                <img
                  src={previewUrl}
                  alt="Скріншот новини"
                  className="h-14 w-14 object-cover rounded border border-neutral-700"
                />
              )}
              <div className="min-w-0 flex-1 pr-8">
                <p className="text-white text-sm truncate">Скріншот зі буфера обміну</p>
                <p className="text-neutral-500 text-xs mt-0.5">
                  {(screenshotFile.size / 1024).toFixed(0)} КБ · натисніть ✕ щоб прибрати
                </p>
              </div>
              <button
                type="button"
                onClick={() => setScreenshotFile(null)}
                className="absolute top-2 right-2 text-neutral-400 hover:text-red-500 text-lg leading-none"
                aria-label="Прибрати скріншот"
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {showModeToggles && (
          <div className="mt-6 space-y-3">
            <p className="text-sm text-neutral-500 text-center md:text-left">
              Контекст перевірки (лише для посилання)
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center sm:justify-start">
              <label className="flex items-center gap-3 cursor-pointer rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 hover:border-red-500/60 has-[:checked]:border-red-500">
                <input
                  type="radio"
                  name="verificationMode"
                  className="accent-red-600"
                  checked={verificationMode === "publication"}
                  onChange={() => setVerificationMode("publication")}
                />
                <span className="text-white text-sm leading-snug">
                  <span className="font-semibold block">На момент публікації</span>
                  <span className="text-neutral-500 text-xs">
                    Дивимось на новину так, ніби ми в день її виходу (стара стаття не
                    стає неправдивою лише тому, що обставини змінилися)
                  </span>
                </span>
              </label>
              <label className="flex items-center gap-3 cursor-pointer rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 hover:border-red-500/60 has-[:checked]:border-red-500">
                <input
                  type="radio"
                  name="verificationMode"
                  className="accent-red-600"
                  checked={verificationMode === "today"}
                  onChange={() => setVerificationMode("today")}
                />
                <span className="text-white text-sm leading-snug">
                  <span className="font-semibold block">На сьогоднішній день</span>
                  <span className="text-neutral-500 text-xs">
                    Дивимось на новину очима сьогодні: застарілі формулювання можуть
                    виглядати ризикованіше
                  </span>
                </span>
              </label>
            </div>
          </div>
        )}

        <div className="mt-6">
          <button
            type="button"
            onClick={onAnalyze}
            disabled={isLoading || (!inputUrl.trim() && !screenshotFile)}
            className="w-full px-8 py-4 bg-red-600 hover:bg-red-700 disabled:bg-neutral-700 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-all duration-200 transform hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-red-500 disabled:opacity-50"
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <circle cx="12" cy="12" r="10" strokeWidth="2" opacity="0.25" />
                  <path strokeWidth="2" d="M4 12a8 8 0 018-8" />
                </svg>
                Аналізую…
              </span>
            ) : screenshotFile ? (
              "Перевірити скрін"
            ) : (
              "Analyze News"
            )}
          </button>
        </div>

        <p className="text-center text-neutral-600 text-sm mt-6">
          Скрін: Win+Shift+S → клацніть у поле → Ctrl+V. Ctrl+Enter — запуск аналізу.
        </p>
      </div>
    </section>
  )
}
