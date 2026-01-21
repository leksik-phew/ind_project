import os
import csv
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ----------------------------
# Config
# ----------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("edu-ai-bot-ollama")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen2.5:7b")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")

CSV_PATH = Path(os.getenv("CSV_PATH", "ratings.csv"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

SYSTEM_HINT = (
    "Ты — помощник для учебных задач. "
    "Отвечай кратко и по делу. "
    "Если данных недостаточно — задай 1 уточняющий вопрос."
)

# ----------------------------
# CSV logging
# ----------------------------
CSV_HEADERS = [
    "timestamp_utc",
    "user_id",
    "chat_id",
    "message_id",
    "input_type",
    "user_text",
    "text_model",
    "vision_model",
    "ai_answer",
    "rating",
]

def ensure_csv_exists() -> None:
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

def append_csv_row(row: dict) -> None:
    ensure_csv_exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)

# ----------------------------
# Ollama client helpers
# ----------------------------
class OllamaError(Exception):
    pass

def ollama_generate(
    model: str,
    prompt: str,
    images_b64: Optional[List[str]] = None,
    timeout_sec: int = 180,
) -> str:
    """
    Calls Ollama /api/generate.
    For vision models (e.g., llava), pass images_b64=[<base64 string>, ...].
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if images_b64:
        payload["images"] = images_b64

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=timeout_sec,
        )
    except requests.RequestException as e:
        raise OllamaError(f"Ollama request failed: {e}") from e

    if r.status_code != 200:
        # Попробуем вытащить текст ошибки
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise OllamaError(f"Ollama HTTP {r.status_code}: {err}")

    data = r.json()
    # Обычно ответ в "response"
    return (data.get("response") or "").strip()

def build_prompt(user_text: Optional[str], mode: str) -> str:
    """
    mode: "text" or "vision"
    """
    if mode == "text":
        # Текстовый запрос
        text = (user_text or "").strip()
        if text:
            return f"{SYSTEM_HINT}\n\nЗапрос пользователя:\n{text}"
        return f"{SYSTEM_HINT}\n\nПользователь ничего не написал. Попроси уточнить задачу."

    # Vision (фото)
    caption = (user_text or "").strip()
    if caption:
        return (
            f"{SYSTEM_HINT}\n\n"
            f"Пользователь прислал изображение и подпись:\n{caption}\n\n"
            "Сначала кратко опиши, что видишь на изображении, "
            "потом реши/объясни по запросу из подписи."
        )
    return (
        f"{SYSTEM_HINT}\n\n"
        "Пользователь прислал изображение без подписи. "
        "1) Опиши, что на изображении. "
        "2) Предложи, какую учебную пользу можно из этого извлечь."
    )

# ----------------------------
# Rating UI
# ----------------------------
def rating_keyboard(token: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton("1", callback_data=f"rate|{token}|1"),
        InlineKeyboardButton("2", callback_data=f"rate|{token}|2"),
        InlineKeyboardButton("3", callback_data=f"rate|{token}|3"),
        InlineKeyboardButton("4", callback_data=f"rate|{token}|4"),
        InlineKeyboardButton("5", callback_data=f"rate|{token}|5"),
    ]
    return InlineKeyboardMarkup([buttons])

def make_token(chat_id: int, message_id: int) -> str:
    return f"{chat_id}:{message_id}"

async def send_answer_with_rating(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    input_type: str,
    user_text: str,
    ai_answer: str,
) -> None:
    sent = await update.message.reply_text(ai_answer)

    token = make_token(sent.chat_id, sent.message_id)
    context.application.bot_data.setdefault("pending", {})[token] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "user_id": update.effective_user.id if update.effective_user else None,
        "chat_id": sent.chat_id,
        "message_id": sent.message_id,
        "input_type": input_type,
        "user_text": user_text,
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "ai_answer": ai_answer,
    }

    await update.message.reply_text(
        "Оцени ответ по шкале 1–5:",
        reply_markup=rating_keyboard(token),
    )

# ----------------------------
# Telegram handlers
# ----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я локальный AI-бот (Ollama).\n\n"
        "• Отправь текст — отвечу\n"
        "• Отправь фото (+ подпись) — опишу/решу по фото\n\n"
        "После ответа появится оценка 1–5, а результат сохранится в CSV."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды: /start /help\n\n"
        "Как пользоваться:\n"
        "1) Текстом: отправь вопрос\n"
        "2) Фото: отправь фото и (желательно) подпись, что сделать\n\n"
        "Важно:\n"
        "- Ollama должна быть запущена\n"
        f"- Модели: TEXT_MODEL={TEXT_MODEL}, VISION_MODEL={VISION_MODEL}"
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text or ""
    await update.message.chat.send_action(action="typing")

    prompt = build_prompt(user_text, mode="text")

    try:
        answer = ollama_generate(TEXT_MODEL, prompt, images_b64=None, timeout_sec=180)
        if not answer:
            answer = "Пустой ответ от модели. Попробуй переформулировать вопрос."
        await send_answer_with_rating(
            update, context,
            input_type="text",
            user_text=user_text,
            ai_answer=answer,
        )
    except OllamaError as e:
        log.exception("Ollama text error")
        await update.message.reply_text(
            "Ошибка при обращении к локальной модели (Ollama).\n"
            f"{e}\n\n"
            "Проверь, что Ollama запущена и модель скачана."
        )
    except Exception as e:
        log.exception("Unexpected text error")
        await update.message.reply_text(f"Неожиданная ошибка: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo = update.message.photo[-1]
    caption = update.message.caption or ""
    await update.message.chat.send_action(action="typing")

    try:
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        # Ollama ждёт base64 без "data:image/..;base64,"
        img_b64 = base64.b64encode(bytes(image_bytes)).decode("utf-8")

        prompt = build_prompt(caption, mode="vision")

        answer = ollama_generate(
            VISION_MODEL,
            prompt,
            images_b64=[img_b64],
            timeout_sec=240,
        )
        if not answer:
            answer = "Пустой ответ от vision-модели. Попробуй добавить подпись к фото с задачей."

        await send_answer_with_rating(
            update, context,
            input_type="photo",
            user_text=caption,
            ai_answer=answer,
        )
    except OllamaError as e:
        log.exception("Ollama vision error")
        await update.message.reply_text(
            "Ошибка при обработке фото через Ollama.\n"
            f"{e}\n\n"
            "Проверь, что vision-модель скачана (например, llava) и Ollama запущена."
        )
    except Exception as e:
        log.exception("Unexpected photo error")
        await update.message.reply_text(f"Неожиданная ошибка при фото: {e}")

async def handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    try:
        parts = (query.data or "").split("|")
        if len(parts) != 3 or parts[0] != "rate":
            await query.edit_message_text("Некорректные данные оценки.")
            return

        token, rating_str = parts[1], parts[2]
        if rating_str not in {"1", "2", "3", "4", "5"}:
            await query.edit_message_text("Оценка должна быть от 1 до 5.")
            return

        pending = context.application.bot_data.get("pending", {})
        record = pending.get(token)
        if not record:
            await query.edit_message_text(
                "Не нашёл, к какому ответу относится оценка (возможно, уже сохранено)."
            )
            return

        row = {**record, "rating": int(rating_str)}
        append_csv_row(row)

        del pending[token]
        await query.edit_message_text(f"Спасибо! Оценка: {rating_str}/5 ✅")

    except Exception as e:
        log.exception("Rating handler error")
        try:
            await query.edit_message_text(f"Ошибка при сохранении оценки: {e}")
        except Exception:
            pass

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ensure_csv_exists()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(CallbackQueryHandler(handle_rating, pattern=r"^rate\|"))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    log.info("Bot started (Ollama)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
