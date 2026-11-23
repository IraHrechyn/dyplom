from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from telegram import InputFile, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

BOT_DIR = Path(__file__).resolve().parent
ROOT = BOT_DIR.parent
PY_DIR = ROOT / "python"
for p in (str(BOT_DIR), str(PY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

from analyze_url import (
    VERIFICATION_PUBLICATION,
    VERIFICATION_TODAY,
    _load_sklearn,
    analyze_raw_text,
    analyze_url,
)
from chart_image import render_conclusion_chart
from factcheck_client import enrich_like_web, factcheck_api_url
from news_article_extractor import NewsExtractionError
from ocr_screenshot import OcrError, extract_text_from_image

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("telegram_bot")

URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
TG_LIMIT = 3900
MAX_IMAGE_BYTES = 12 * 1024 * 1024

def _is_url(text: str) -> bool:
    return bool(URL_RE.match(text.strip()))

def _clip(s: str, n: int = 280) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"

def _esc(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def _format_facts(title: str, items: list[dict], *, empty_hint: str) -> str:
    if not items:
        return f"<b>{_esc(title)}</b>\n<i>{_esc(empty_hint)}</i>\n"
    lines = [f"<b>{_esc(title)}</b>"]
    for i, fr in enumerate(items, 1):
        conf = fr.get("confidence")
        conf_pct = int(round(float(conf) * 100)) if conf is not None else None
        claim = _clip(str(fr.get("claim") or ""), 220)
        meta = f" ({conf_pct}%)" if conf_pct is not None else ""
        provider = str(fr.get("provider") or "").strip()
        if provider:
            meta = f"{meta} [{provider}]" if meta else f" [{provider}]"
        lines.append(f"{i}. {_esc(claim)}{_esc(meta)}")
        citations = fr.get("citations") if isinstance(fr.get("citations"), list) else []
        for cit in citations[:2]:
            if not isinstance(cit, dict):
                continue
            url = str(cit.get("url") or "").strip()
            quote = str(cit.get("quote") or "").strip()
            if url:
                lines.append(f"   🔗 {_esc(_clip(url, 120))}")
            if quote:
                lines.append(f"   <i>«{_esc(_clip(quote, 160))}»</i>")
        note = (fr.get("note") or "").strip()
        if note:
            lines.append(f"   <i>{_esc(_clip(note, 180))}</i>")
    return "\n".join(lines) + "\n"

def format_result(result: dict, *, input_kind: str, ocr_preview: str | None = None) -> str:
    mode = result.get("modeTitle") or ""
    consensus = result.get("factCheckConsensus") or {}
    true_facts = result.get("trueFacts") or []
    false_facts = result.get("falseFacts") or []
    combined = result.get("combined") or {}

    kind_labels = {
        "url": "посилання",
        "plain_text": "текст",
        "screenshot": "скріншот",
    }
    kind_label = kind_labels.get(input_kind, input_kind)
    parts = [
        f"<b>Результати аналізу</b> ({_esc(kind_label)})",
    ]
    if mode:
        parts.append(f"📌 {_esc(mode)}")

    if ocr_preview:
        parts.append("")
        parts.append(f"<b>Зчитано зі скріна</b>\n<i>«{_esc(_clip(ocr_preview, 500))}»</i>")

    note = (combined.get("note") or "").strip()
    if note:
        parts.append("")
        parts.append(f"<i>{_esc(note)}</i>")

    summary = ""
    analysis = ""
    if isinstance(consensus, dict):
        summary = str(consensus.get("summary") or "").strip()
        analysis = str(consensus.get("analysis") or "").strip()
    if summary or analysis:
        parts.append("")
        parts.append("<b>Висновок фактчеку</b>")
        if isinstance(consensus, dict):
            src = " / ".join(
                x for x in [str(consensus.get("provider") or "").strip(), str(consensus.get("model") or "").strip()] if x
            )
            if src:
                parts.append(f"<i>Джерело: {_esc(src)}</i>")
        if summary:
            parts.append(_esc(summary))
        if analysis and analysis != summary:
            parts.append("")
            parts.append(_esc(_clip(analysis, 1200)))
    elif result.get("factCheckError"):
        parts.append("")
        parts.append("<b>Фактчек недоступний</b>")
        parts.append(_esc(_clip(str(result.get("factCheckError")), 400)))

    if true_facts:
        parts.append("")
        parts.append(_format_facts("✅ Правдиві факти", true_facts, empty_hint=""))
    if false_facts:
        parts.append("")
        parts.append(_format_facts("❌ Неправдиві факти", false_facts, empty_hint=""))
    if not true_facts and not false_facts:
        parts.append("")
        parts.append("<i>Ключових підтверджених чи спростованих фактів не виділено.</i>")

    text = "\n".join(parts).strip()
    if len(text) > TG_LIMIT:
        text = text[: TG_LIMIT - 1] + "…"
    return text

async def _reply_analysis(
    update: Update,
    status,
    result: dict,
    *,
    input_kind: str,
    ocr_preview: str | None = None,
) -> None:
    if result.get("error"):
        await status.edit_text(f"Помилка: {result['error']}")
        return

    mb = result.get("modelBreakdown") or {}
    log.info(
        "[стиль] lang=%s model=%s risk=%s%% truth=%s%%",
        mb.get("language"),
        mb.get("activeModel"),
        result.get("fakePercentage"),
        result.get("truthPercentage"),
    )
    await status.edit_text("⏳ Стиль оцінено. Запускаю фактчек…")
    enriched = await asyncio.to_thread(enrich_like_web, result)
    if enriched.get("factCheckError"):
        log.warning("[фактчек] %s", enriched.get("factCheckError"))
    else:
        log.info(
            "[фактчек] providers=%s claims_true=%s claims_false=%s",
            [fc.get("provider") for fc in (enriched.get("factChecks") or [])],
            len(enriched.get("trueFacts") or []),
            len(enriched.get("falseFacts") or []),
        )
    combined = enriched.get("combined") or {}
    log.info(
        "[soft vote] risk=%s%% truth=%s%% note=%s",
        combined.get("combinedFakePercent", enriched.get("fakePercentage")),
        combined.get("combinedTruthPercent", enriched.get("truthPercentage")),
        combined.get("note"),
    )

    truth = float(enriched.get("truthPercentage") or 0)
    risk = float(enriched.get("fakePercentage") or 0)
    png = await asyncio.to_thread(render_conclusion_chart, truth, risk)
    reply = format_result(enriched, input_kind=input_kind, ocr_preview=ocr_preview)

    await status.delete()
    await update.message.reply_photo(
        photo=InputFile(io.BytesIO(png), filename="conclusion.png"),
    )
    await update.message.reply_text(
        reply,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привіт! Надішли <b>посилання</b>, <b>текст</b>, <b>відео з підписом</b> "
        "або <b>скріншот</b> новини.\n\n"
        "Я поверну той самий аналіз, що й сайт:\n"
        "• діаграму надійності / ризику\n"
        "• висновок фактчеку\n"
        "• ключові правдиві та неправдиві факти\n\n"
        "Команди:\n"
        "/start — довідка\n"
        "/help — як користуватись",
        parse_mode=ParseMode.HTML,
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "1) Встав URL виду https://…\n"
        "2) Або встав текст новини (краще від 2–3 речень)\n"
        "3) Або перешли пост з відео — перевірю текстовий підпис\n"
        "4) Або надішли скріншот новини (фото / файл-картинка)\n"
        "5) Зачекай — аналіз може зайняти 1–3 хвилини (стиль + фактчек)\n\n"
        "Для посилання — режим «на момент публікації», для тексту / відео / скріна — «на сьогодні».\n"
        "Потрібен запущений веб-сервер (npm run dev).",
    )

async def _analyze_text_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    raw: str,
    *,
    status_text: str = "⏳ Аналізую… Це може зайняти трохи часу.",
    input_kind: str = "plain_text",
    ocr_preview: str | None = None,
) -> None:
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    status = await update.message.reply_text(status_text)

    try:
        text = (raw or "").strip()
        if _is_url(text):
            result = await asyncio.to_thread(
                analyze_url,
                text,
                15,
                VERIFICATION_PUBLICATION,
            )
            kind = "url"
        else:
            if len(text) < 15:
                await status.edit_text(
                    "Текст занадто короткий. Надішли хоча б кілька речень, URL, "
                    "відео з підписом або скріншот."
                )
                return
            result = await asyncio.to_thread(
                analyze_raw_text,
                text,
                VERIFICATION_TODAY,
            )
            kind = input_kind

        await _reply_analysis(
            update,
            status,
            result,
            input_kind=kind,
            ocr_preview=ocr_preview,
        )
    except NewsExtractionError as e:
        log.warning("extraction/analyze error: %s", e)
        try:
            await status.edit_text(f"Не вдалося проаналізувати: {e}")
        except Exception:
            await update.message.reply_text(f"Не вдалося проаналізувати: {e}")
    except Exception as e:
        log.exception("bot handler failed")
        try:
            await status.edit_text(f"Сталася помилка: {e}")
        except Exception:
            await update.message.reply_text(f"Сталася помилка: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    raw = update.message.text.strip()
    if not raw or raw.startswith("/"):
        return

    await _analyze_text_message(update, context, raw)

async def handle_captioned_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    caption = (update.message.caption or "").strip()
    if len(caption) < 15:
        await update.message.reply_text(
            "Бачу відео, але підпису майже немає.\n\n"
            "Надішли новину з текстовим підписом, або скопіюй текст / посилання окремим повідомленням.\n"
            "(Саме відео зараз не розпізнається — лише текст під ним.)"
        )
        return

    await _analyze_text_message(
        update,
        context,
        caption,
        status_text="🎬 Беру підпис до відео й аналізую…",
        input_kind="plain_text",
        ocr_preview=caption,
    )

async def _download_image_bytes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bytes:
    msg = update.message
    file_id = None
    if msg.photo:
        file_id = msg.photo[-1].file_id
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        file_id = msg.document.file_id
        if msg.document.file_size and msg.document.file_size > MAX_IMAGE_BYTES:
            raise OcrError("Файл занадто великий (макс. ~12 МБ). Надішли стиснуте фото.")
    else:
        raise OcrError("Немає зображення в повідомленні")

    tg_file = await context.bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(buf)
    data = buf.getvalue()
    if not data:
        raise OcrError("Не вдалося завантажити зображення з Telegram")
    if len(data) > MAX_IMAGE_BYTES:
        raise OcrError("Зображення занадто велике. Надішли стиснуте фото.")
    return data

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    status = await update.message.reply_text("🖼️ Зчитую текст зі скріна…")

    try:
        image_bytes = await _download_image_bytes(update, context)
        await status.edit_text("🖼️ Зчитую текст зі скріна…\n⏳ Далі перевіряю надійність…")

        text = await asyncio.to_thread(extract_text_from_image, image_bytes)
        if len(text.strip()) < 15:
            await status.edit_text(
                "На скріні замало тексту. Зробіть чіткіший скріншот або надішліть текст / посилання."
            )
            return

        result = await asyncio.to_thread(
            analyze_raw_text,
            text,
            VERIFICATION_TODAY,
        )
        await _reply_analysis(
            update,
            status,
            result,
            input_kind="screenshot",
            ocr_preview=text,
        )

    except OcrError as e:
        log.warning("ocr error: %s", e)
        try:
            await status.edit_text(f"Не вдалося зчитати скрін: {e}")
        except Exception:
            await update.message.reply_text(f"Не вдалося зчитати скрін: {e}")
    except NewsExtractionError as e:
        log.warning("analyze after ocr error: %s", e)
        try:
            await status.edit_text(f"Текст зі скріна зчитано, але аналіз не вдався: {e}")
        except Exception:
            await update.message.reply_text(f"Текст зі скріна зчитано, але аналіз не вдався: {e}")
    except Exception as e:
        log.exception("photo handler failed")
        try:
            await status.edit_text(f"Сталася помилка: {e}")
        except Exception:
            await update.message.reply_text(f"Сталася помилка: {e}")

def main() -> None:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "Не знайдено TELEGRAM_BOT_TOKEN.\n"
            "Додайте його у файл .env.local у корені проєкту, наприклад:\n"
            "TELEGRAM_BOT_TOKEN=123456:ABC..."
        )

    try:
        _load_sklearn()
        log.info("UA TF-IDF модель прогріто в пам'яті")
    except Exception as e:
        log.warning("Не вдалося прогріти UA-модель заздалегідь: %s", e)

    log.info("FACTCHECK_API_URL=%s", factcheck_api_url())

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_photo))
    app.add_handler(
        MessageHandler(
            filters.VIDEO | filters.VIDEO_NOTE | filters.ANIMATION | filters.Document.VIDEO,
            handle_captioned_media,
        )
    )

    log.info("Telegram bot starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
