import requests
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import ParseMode
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types.message import ContentType
from aiogram.utils import executor
from aiogram.utils.emoji import emojize
from aiogram.utils.markdown import text, bold, italic, code

from config import TOKEN

from PostalService.src.infer import *
from PostalService.src.recognizeTypewrittenText import recognize_typewritten_text

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

error_process_messages = [
    'Я не знаю что с этим делать :man_shrugging:'
]

prediction_process_messages = [
    'Мне кажется, здесь написано \n\n@',
    'Похоже на \n\n@',
    'Кажется, тут написано: \n\n@',
    'Думаю, там написано \n\n@',
    '@\n\n Угадал?'
]

demo_process_messages = [
    'раз сам писать не хочешь, поищу у себя что-нибудь...\nнашел!',
    'понимаю, не у всех есть бумага с ручкой под рукой\nдержи',
    'передаю запрос на спутник... :thinking_face:',
    'устанавливаю связь с космосом... :thinking_face:',
]

char_list = list('абвгдежзийклмнопрстуфхцчшщъыьэюя ')
decoder_mapping = {'bestpath': DecoderType.BestPath,
                   'beamsearch': DecoderType.BeamSearch,
                   'wordbeamsearch': DecoderType.WordBeamSearch}
decoder_type = decoder_mapping["beamsearch"]

model = Model(char_list, decoder_type, must_restore=True)

img = None


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    start_kb = ReplyKeyboardMarkup(
        resize_keyboard=True, one_time_keyboard=True
    ).add(KeyboardButton('/help'))

    await message.reply(text('Привет! 👋\nЯ - бот, который любит распознавать текст. 😏'
                             '\n\nИспользуй /help, чтобы узнать список доступных команд!'),
                        reply_markup=start_kb, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    demo_b = KeyboardButton('/demo')
    info_b = KeyboardButton('/info')
    help_kb = ReplyKeyboardMarkup(
        resize_keyboard=True, one_time_keyboard=False
    ).add(demo_b).add(info_b)

    msg = emojize(
        text(
            bold('❗❗❗ На данный момент бот находится в бета') + '-' + bold('тестировании') + '.' +
            bold(' Планируется сделать полную работу в автоматическом режиме') + '.' +
            bold(' Сейчас можно проверить работоспособность каждого модуля по отдельности❗❗❗'),

            '\n\nДля распознавания ' + bold('машинописного') + ' (' + bold('печатного') + ')' + ' текста:',
            '• Присылай фотографию почтового отправления, а я попробую узнать, что там написано',

            '\nДля распознавания ' + italic('рукописного') + ' текста:',
            '• Присылай яркое и контрастное фото, с текстом в одну строку, а я попробую угадать, что ты там написал',
            bold('\n\nДоступны следующие команды:'),
            '/demo - Проверить нейросеть на случайном изображении (9 вариаций)',
            '/info - Подробная информация о проекте',
            sep='\n'))

    await bot.send_message(message.from_user.id, msg, parse_mode=ParseMode.MARKDOWN, reply_markup=help_kb)


@dp.message_handler(commands=['demo'])
async def process_photo_command(msg: types.Message):
    files = os.listdir(path="./Demonstration")

    rand = random.randint(1, len(files))

    global img

    with open(f"./Demonstration/demo_{rand}.png", "rb") as image:
        img = image

        caption = None

        if rand == 1 or rand == 3 or rand == 9:
            recognized = recognize_typewritten_text(img)
            i = random.randint(0, len(prediction_process_messages) - 1)
            caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@',
                                                                                                ' '.join(recognized))
        if rand == 2 or rand == 4 or rand == 5 or rand == 6 or rand == 7 or rand == 8:
            recognized, recognized_corrected = bot_run_model(model, img)
            i = random.randint(0, len(prediction_process_messages) - 1)
            caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@',
                                                                                                recognized_corrected)

        await bot.send_photo(msg.from_user.id, types.InputFile(f"./Demonstration/demo_{rand}.png", "rb"),
                             caption=emojize(caption))


def get_path(hash_name):
    link = 'https://api.telegram.org/bot<TOKEN>/getFile?file_id=<FILE_ID>'.replace('<TOKEN>', TOKEN)
    path = 'https://api.telegram.org/file/bot<TOKEN>/<FILE_PATH>'.replace('<TOKEN>', TOKEN)
    link = link.replace('<FILE_ID>', hash_name)
    r = requests.get(url=link)
    file_path = r.json()['result']['file_path']
    file_path = path.replace('<FILE_PATH>', file_path)
    return file_path


@dp.message_handler(commands=['info'])
async def process_info_command(msg: types.Message):
    info = text('Создано командой MasterMinds' + '\nБудет заполнено к защите.')
    await bot.send_message(msg.from_user.id, info)


@dp.callback_query_handler(text="typewritten_text")
async def recognize_text(call: types.CallbackQuery):
    recognized = recognize_typewritten_text(img)
    i = random.randint(0, len(prediction_process_messages) - 1)
    caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@', ' '.join(recognized))
    await bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                text=caption, parse_mode=ParseMode.MARKDOWN)


@dp.callback_query_handler(text="handwritten_text")
async def recognize_text(call: types.CallbackQuery):
    # try:
    # try:
    recognized, recognized_corrected = bot_run_model(model, img)
    print(recognized, recognized_corrected)

    i = random.randint(0, len(prediction_process_messages) - 1)
    caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@', recognized_corrected)

    await bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                text=caption, parse_mode=ParseMode.MARKDOWN)

@dp.message_handler(content_types=['photo'])
async def echo_img(msg: types.Message):
    img_hash = msg.photo[-1]['file_id']
    url = get_path(img_hash)
    global img
    img = requests.get(url, stream=True).raw

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton(text="Распознать машинописный текст",
                                            callback_data="typewritten_text"))
    keyboard.add(types.InlineKeyboardButton(text="Распознать рукописный текст",
                                            callback_data="handwritten_text"))
    await msg.reply("Что мне сделать с этим?", reply_markup=keyboard)


@dp.message_handler(content_types=ContentType.ANY)
async def unknown_message(msg: types.Message):
    i = random.randint(0, len(error_process_messages) - 1)
    message_text = text(emojize(error_process_messages[i]),
                        emojize(text(italic('\nЯ просто напомню,'), 'что есть')),
                        code('команда'), '/help')
    await msg.reply(message_text, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    executor.start_polling(dp)
