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

from PostalService.src.recognizeHandwriteText import *
from PostalService.src.recognizeTypewrittenText import recognize_typewritten_text
# from PostalService.src.text_classificator import text_type_classification
from PostalService.src.recognizeAllTypesTexts import recognize_all_types_of_written_text

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

num_demo_img = 0


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    start_kb = ReplyKeyboardMarkup(
        resize_keyboard=True, one_time_keyboard=True
    ).add(KeyboardButton('/help'))

    await message.reply(text('Привет! 👋\nЯ - бот, который любит распознавать текст текст с почтовых отправлений. 😏'
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
            bold('❗❗❗Помимо автоматического распознавания как Рукописного, так и Машинописного текста, '
                 'можно проверить работоспособность каждого модуля по отдельности❗❗❗'),

            '\n\nДля распознавания текста ' + bold('ЛЮБОГО типа') + ':',
            '• Присылай фотографию почтового отправления, а я попробую узнать, что там написано',

            '\n\nДля распознавания ' + bold('ТОЛЬКО машинописного') + ' (' + bold('печатного') + ')' + ' текста:',
            '• Отправляй фотографию почтового отправления, без рукописного текста, '
            'иначе я запутаюсь :smiling_face_with_tear:',

            '\nДля распознавания ' + italic('ТОЛЬКО рукописного') + ' текста:',
            '• Присылай яркое и контрастное фото, с текстом в одну строку, а я попробую угадать, что ты там написал. ',

            bold('\n\nДоступны следующие команды:'),
            '/demo - Проверить нейросеть на случайном изображении (9 вариаций)',
            '/info - Подробная информация о проекте',
            sep='\n'))

    await bot.send_message(message.from_user.id, msg, parse_mode=ParseMode.MARKDOWN, reply_markup=help_kb)


@dp.message_handler(commands=['demo'])
async def process_photo_command(msg: types.Message):

    global num_demo_img

    num_demo_img += 1

    if num_demo_img >= 9:
        num_demo_img = 0

    global img

    img_path = f"./Demonstration/demo_{num_demo_img}.png"

    with open(img_path, "rb") as image:

        img = image

        caption = None

        if num_demo_img == 0 or num_demo_img == 2 or num_demo_img == 8:
            recognized = recognize_typewritten_text(img)
            i = random.randint(0, len(prediction_process_messages) - 1)
            caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@',
                                                                                                ' '.join(recognized))
        if num_demo_img == 1 or num_demo_img == 3 or num_demo_img == 4 or num_demo_img == 5 or num_demo_img == 7:
            recognized, recognized_corrected = bot_run_model(model, img)
            i = random.randint(0, len(prediction_process_messages) - 1)
            caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@',
                                                                                                recognized_corrected)
        if num_demo_img == 6:
            recognized = recognize_all_types_of_written_text(img, model)
            i = random.randint(0, len(prediction_process_messages) - 1)
            caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@',
                                                                                                ' '.join(recognized))

        await bot.send_photo(msg.from_user.id, types.InputFile(img_path, "rb"),
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
    info = text('Создано командой MasterMinds' + '\n\nО проекте: project.ai-info.ru/teams/masterminds' +
                "\nGitHub: https://github.com/GamerZombar/PostalService" +
                "\n\nДля просмотра инструкции - команда /help")
    await bot.send_message(msg.from_user.id, info)


@dp.callback_query_handler(text="typewritten_text")
async def recognize_text(call: types.CallbackQuery):
    recognized = recognize_typewritten_text(img)
    i = random.randint(0, len(prediction_process_messages) - 1)
    caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@', ' '.join(recognized))
    await bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                text=caption)


@dp.callback_query_handler(text="handwritten_text")
async def recognize_text(call: types.CallbackQuery):
    recognized, recognized_corrected = bot_run_model(model, img)
    print(recognized, recognized_corrected)

    i = random.randint(0, len(prediction_process_messages) - 1)
    caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@', recognized_corrected)

    await bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                text=caption)


@dp.callback_query_handler(text="all_types_text")
async def recognize_text(call: types.CallbackQuery):
    print("\nНачинаю работу. Сообщение от " + str(call.message.date))
    recognized = recognize_all_types_of_written_text(img, model)

    i = random.randint(0, len(prediction_process_messages) - 1)
    caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@', ' '.join(recognized))
    await bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                text=caption)


@dp.message_handler(content_types=['photo'])
async def echo_img(msg: types.Message):
    img_hash = msg.photo[-1]['file_id']
    url = get_path(img_hash)

    global img
    img = requests.get(url, stream=True).raw


    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.KeyboardButton(text="Только машинописный текст", callback_data="typewritten_text"),
                 types.KeyboardButton(text="Только рукописный текст", callback_data="handwritten_text"))

    keyboard.add(types.InlineKeyboardButton(text="Текст ЛЮБОГО типа",
                                            callback_data="all_types_text"))
    await msg.reply("Как вы хотите это распознать?", reply_markup=keyboard)


@dp.message_handler(content_types=ContentType.ANY)
async def unknown_message(msg: types.Message):
    i = random.randint(0, len(error_process_messages) - 1)
    message_text = text(emojize(error_process_messages[i]),
                        emojize(text(italic('\nЯ просто напомню,'), 'что есть')),
                        code('команда'), '/help')
    await msg.reply(message_text, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    executor.start_polling(dp)
