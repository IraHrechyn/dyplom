"use client"

import { useState } from "react"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

const FEATURES = [
  {
    step: "01",
    title: "Посилання, текст або скрін",
    text: "Вставте URL, текст статті або скріншот через Ctrl+V — система зчитає зміст автоматично.",
  },
  {
    step: "02",
    title: "Оцінка стилю",
    text: "Моделі машинного навчання оцінюють, наскільки текст схожий на фейк за стилем і формулюваннями.",
  },
  {
    step: "03",
    title: "Фактчек із джерелами",
    text: "Фактчек звіряє ключові твердження з відкритими джерелами: логіка фактів, ознаки фейку та незалежні підтвердження.",
  },
  {
    step: "04",
    title: "Зрозумілий результат",
    text: "Діаграма ризику, висновок фактчеку, ключові факти та матеріали на ту саму тему.",
  },
] as const

export default function HeroSection() {
  const [open, setOpen] = useState(false)

  const scrollToAnalyze = () => {
    setOpen(false)
    window.setTimeout(() => {
      const el = document.getElementById("analyze")
      if (!el) return
      el.scrollIntoView({ behavior: "smooth", block: "start" })
      window.setTimeout(() => {
        const input = document.getElementById("analyze-input") as HTMLTextAreaElement | null
        input?.focus()
      }, 450)
    }, 150)
  }

  return (
    <section className="w-full min-h-screen flex items-center justify-center relative overflow-hidden bg-gradient-to-b from-black to-neutral-950 newspaper-texture">
      <div className="absolute inset-0 opacity-10">
        <div
          className="w-full h-full"
          style={{
            backgroundImage:
              "linear-gradient(0deg, transparent 24%, rgba(255, 255, 255, 0.05) 25%, rgba(255, 255, 255, 0.05) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, 0.05) 75%, rgba(255, 255, 255, 0.05) 76%, transparent 77%, transparent), linear-gradient(90deg, transparent 24%, rgba(255, 255, 255, 0.05) 25%, rgba(255, 255, 255, 0.05) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, 0.05) 75%, rgba(255, 255, 255, 0.05) 76%, transparent 77%, transparent)",
            backgroundSize: "50px 50px",
          }}
        />
      </div>

      <div className="relative z-10 text-center px-4 max-w-4xl mx-auto">
        <div className="mb-6">
          <span className="inline-block text-xs font-mono tracking-widest text-red-500 mb-6 uppercase">
            Deep Learning Detection System
          </span>
        </div>

        <h1 className="text-6xl md:text-7xl font-bold mb-8 leading-tight text-white text-balance">
          <span className="text-red-500">Fake News</span> Detector
        </h1>

        <p className="text-lg md:text-xl text-neutral-400 mb-12 max-w-2xl mx-auto leading-relaxed text-pretty">
          Аналізуйте новини та пости: виявляйте оманливий контент, перевіряйте факти
          й розкривайте маніпуляції з науковою точністю.
        </p>

        <div className="flex justify-center gap-4">
          <button
            type="button"
            onClick={scrollToAnalyze}
            className="px-8 py-3 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg transition-all duration-200 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-red-500"
          >
            Почати аналіз
          </button>

          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
              <button
                type="button"
                className="px-8 py-3 border-2 border-white hover:bg-white hover:text-black text-white font-semibold rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-white"
              >
                Дізнатися більше
              </button>
            </DialogTrigger>

            <DialogContent
              showCloseButton={false}
              className="max-w-2xl border-neutral-800 bg-neutral-950 p-0 text-white shadow-2xl shadow-red-950/20 sm:max-w-2xl overflow-hidden"
            >
              <div className="h-1 w-full bg-gradient-to-r from-transparent via-red-500 to-transparent" />

              <div className="px-6 pt-6 pb-2 sm:px-8">
                <DialogHeader className="gap-3 text-left">
                  <p className="text-xs font-mono uppercase tracking-[0.2em] text-red-500">
                    Про сервіс
                  </p>
                  <DialogTitle className="text-2xl sm:text-3xl font-bold text-white leading-tight">
                    Як працює <span className="text-red-500">Fake News Detector</span>
                  </DialogTitle>
                  <DialogDescription className="text-neutral-400 text-base leading-relaxed">
                    Це інструмент для швидкої перевірки новин: ви даєте посилання, текст або скріншот —
                    система оцінює ризик фейку моделями машинного навчання та додатково звіряє
                    ключові твердження з відкритими джерелами.
                  </DialogDescription>
                </DialogHeader>
              </div>

              <div className="px-6 sm:px-8 py-4 space-y-3 max-h-[50vh] overflow-y-auto">
                {FEATURES.map((item) => (
                  <div
                    key={item.step}
                    className="flex gap-4 rounded-xl border border-neutral-800 bg-black/60 px-4 py-3.5"
                  >
                    <span className="shrink-0 font-mono text-sm text-red-500 pt-0.5">{item.step}</span>
                    <div>
                      <p className="text-white font-semibold mb-1">{item.title}</p>
                      <p className="text-sm text-neutral-400 leading-relaxed">{item.text}</p>
                    </div>
                  </div>
                ))}

                <div className="rounded-xl border border-red-900/40 bg-red-950/20 px-4 py-3.5">
                  <p className="text-sm text-neutral-300 leading-relaxed">
                    <span className="text-red-400 font-semibold">Важливо:</span> відсотки показують
                    оцінку ризику за стилем і доступними доказами, а не юридичний вердикт.
                    Завжди перевіряйте критичні новини в офіційних джерелах.
                  </p>
                </div>
              </div>

              <DialogFooter className="border-t border-neutral-800 bg-black/40 px-6 py-4 sm:px-8 gap-3 sm:justify-between">
                <DialogClose asChild>
                  <button
                    type="button"
                    className="px-5 py-2.5 border border-neutral-700 hover:border-neutral-500 text-neutral-300 hover:text-white rounded-lg transition-colors"
                  >
                    Закрити
                  </button>
                </DialogClose>
                <button
                  type="button"
                  onClick={scrollToAnalyze}
                  className="px-5 py-2.5 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-red-500"
                >
                  Почати аналіз
                </button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-red-500 to-transparent" />
    </section>
  )
}
