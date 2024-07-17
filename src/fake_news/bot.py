from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from eval import *

# Токен, который вы получили от BotFather
TOKEN = '7469570269:AAH4FWSEOSXWMcNI1MiBHxT3a5KxdqY5Hfc'


def start(update: Update, context: CallbackContext) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    update.message.reply_text('Привет! Я твой Телеграм-бот. Отправь мне текст новости и я проверю ее на фейковость.')


def handle_message(update: Update, context: CallbackContext) -> None:
    """Обрабатывает текстовые сообщения."""
    # Получаем текст сообщения
    text = update.message.text
    newtext = predict(text)
    # Логируем полученное сообщение (опционально)
    print(f"Получено сообщение: {text}")
    # Добавляем сообщение в историю

    # Отправляем сообщение обратно пользователю
    update.message.reply_text(f"Вы отправили: {newtext}")


def main() -> None:
    """Запуск бота."""
    # Создание экземпляра Updater и передача ему токена вашего бота.
    updater = Updater(TOKEN)

    # Получаем диспетчера для регистрации обработчиков
    dispatcher = updater.dispatcher

    # Регистрируем обработчик команды /start
    dispatcher.add_handler(CommandHandler("start", start))
    # dispatcher.add_handler(CommandHandler("clear", clear_history))

    # Регистрируем обработчик текстовых сообщений
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # Запускаем бота
    updater.start_polling()

    # Ожидаем завершения работы
    updater.idle()


if __name__ == '__main__':
    main()
