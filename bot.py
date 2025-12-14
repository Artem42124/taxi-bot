"""
Telegram Bot for Kremenchuk Taxi Mini App
"""
import asyncio
import os
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo

load_dotenv()

# Configuration
BOT_TOKEN = os.environ.get('BOT_TOKEN')
WEB_APP_URL = os.environ.get('WEB_APP_URL', 'https://taxi-miniapp.preview.emergentagent.com')

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in environment variables")

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_main_keyboard():
    """Create main keyboard with Web App button only"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="\U0001F695 Відкрити застосунок",
            web_app=WebAppInfo(url=WEB_APP_URL)
        )]
    ])
    return keyboard


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Handle /start command"""
    user_name = message.from_user.first_name or "Користувач"
    
    welcome_text = f"""
\U0001F44B Вітаємо, {user_name}!

\U0001F695 **Кременчук Таксі** — ваш надійний сервіс для замовлення поїздок прямо в Telegram!

\u2705 Швидке замовлення
\u2705 Прозорі ціни
\u2705 Надійні водії
\u2705 Працюємо по Кременчуку

Натисніть кнопку нижче, щоб відкрити застосунок:
"""
    
    await message.answer(
        welcome_text,
        parse_mode="Markdown",
        reply_markup=get_main_keyboard()
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Handle /help command"""
    help_text = """
\U0001F4D6 **Довідка**

**Як замовити таксі:**
1. Натисніть кнопку "\U0001F695 Відкрити застосунок"
2. Дозвольте доступ до геолокації
3. Оберіть точку відправлення та призначення
4. Виберіть тариф
5. Підтвердіть замовлення

**Тарифи:**
\U0001F697 Економ — від 45₴
\U0001F699 Комфорт — від 65₴
\U0001F698 Бізнес — від 95₴

**Команди:**
/start — Головне меню
/help — Ця довідка
/contact — Контакти

\U0001F4DE Підтримка: @kremenchuk_taxi_support
"""
    await message.answer(help_text, parse_mode="Markdown")


@dp.message(Command("contact"))
async def cmd_contact(message: types.Message):
    """Handle /contact command"""
    contact_text = """
\U0001F4DE **Контакти**

\U0001F3E2 Кременчук Таксі
\U0001F4CD м. Кременчук, Полтавська область

\U0001F4F1 Телефон: +380 XX XXX XX XX
\U0001F4E7 Email: info@kremenchuk-taxi.ua

\u23F0 Працюємо: 24/7

\U0001F4AC Telegram: @kremenchuk_taxi_support
"""
    await message.answer(contact_text, parse_mode="Markdown")


@dp.message(F.web_app_data)
async def handle_web_app_data(message: types.Message):
    """Handle data from Web App"""
    data = message.web_app_data.data
    logger.info(f"Received web app data: {data}")
    
    await message.answer(
        "\u2705 Дякуємо за використання Кременчук Таксі!",
        reply_markup=get_main_keyboard()
    )


async def main():
    """Start the bot"""
    logger.info("Starting Kremenchuk Taxi Bot...")
    logger.info(f"Web App URL: {WEB_APP_URL}")
    
    # Delete webhook if exists
    await bot.delete_webhook(drop_pending_updates=True)
    
    # Start polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
