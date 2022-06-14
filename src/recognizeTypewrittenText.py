from pathlib import Path
import random
from PIL import Image
import cv2
import pytesseract
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


import numpy as np
from PIL import ImageChops


import threading
import queue


envelope_phrase = ['ФИО', 'Откуда', 'Отправитель', 'Адрес', 'Куда', 'Кому', 'От кого', "Абонент", "Лицевой счет",
                   "Плательщик"]


# def calc_diff(im1, im2):
#     dif = ImageChops.difference(im1, im2)
#     return np.mean(np.array(dif))


# def calc_diff(temp_img, sess):
#     #sess = tf.compat.v1.Session()  # TF session
#
#     def enum_handwritten_pic(im1):
#         im2 = Image.open(random.choice(list(Path(__file__).resolve().parent.parent.joinpath("images").iterdir())))\
#             .convert("L").convert('RGB')  # 1 -- L          # РАНДОМНАЯ КАРТИНКА!!!
#         im2 = im2.resize((300, 300))
#
#         return sess.run(tf.image.ssim(im1, im2, max_val=255))
#
#     temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
#     temp_img = Image.fromarray(temp_img)
#     im1 = temp_img.resize((300, 300))
#
#     res = []
#
#     for i in range(0, 5):
#         res.append(enum_handwritten_pic(im1))
#
#     return np.mean(res)


def recognize_typewritten_text(img_file, sess):
    num_of_valid_data = {}

    result = []

    # windows
    pytesseract.pytesseract.tesseract_cmd = str(Path(__file__).resolve().parent) + "\\Tesseract-OCR\\tesseract.exe"
    tessdata_dir_config = '--tessdata-dir ' + str(Path(__file__).resolve().parent) + "\\Tesseract-OCR\\tessdata"

    arr = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread("EzcyAh436ls.jpg") #EzcyAh436ls.jpg

    image = cv2.resize(image, (0, 0), fx=6, fy=6)

    #image = cv2.bilateralFilter(image,9,75,75)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = thresh

    # вывели
    config = '--psm 6'
    # print(pytesseract.image_to_string(img, config=config, lang='rus'))

    # data = pytesseract.image_to_data(img, lang='rus')

    data = pytesseract.image_to_data(img, lang="rus", config=config,
                                     output_type=pytesseract.Output.DICT)  ## Еще англ. добавь

    data_stock_text = data['text'].copy()

    n_boxes = len(data['level'])

    for i in range(n_boxes):
        data['text'][i] = data['text'][i].lower()
        for phrase in envelope_phrase:
            phrase_lower = phrase.lower()
            if phrase_lower in data['text'][i]:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.putText(img, phrase, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

                num_of_valid_data[phrase] = i
                break

            if phrase_lower.split()[0] in data['text'][i]:
                if phrase_lower.split()[1] in data['text'][i + 1]:
                    (x, y, w, h) = (data['left'][i], data['top'][i],
                                    data['width'][i] + data['width'][i + 1], data['height'][i])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    cv2.putText(img, phrase, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

                    num_of_valid_data[phrase] = i

    for num in num_of_valid_data.items():

        if num is None:
            break

        x = data['width'][num[1]] + data['left'][num[1]]
        y = data['top'][num[1]]

        h = data['height'][num[1]]

        num_current_el = num[1]

        used_el = num[1]

        if not result:
            result.append(num[0] + ": ")
        else:
            result.append('\n' + num[0] + ": ")  # Первое слово, ключ.

        for j in range(n_boxes - num[1] - 2):
            num_current_el += 1

            for i in range(data['left'][num_current_el] + data['width'][num_current_el], data['width'][0], 10):
                current_el_x = data['left'][num_current_el]

                if num_current_el - 1 != num[1]:
                    if current_el_x - (data['left'][used_el] + data['width'][used_el]) >= 500:
                        break

                if data['top'][num_current_el] - 100 <= y <= data['top'][num_current_el] + 100:
                    (x1, y1, w1, h1) = (
                        data['left'][num_current_el], data['top'][num_current_el], data['width'][num_current_el],
                        data['height'][num_current_el])
                    cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), 8)
                    cv2.putText(img, data['text'][num_current_el], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 5,
                                (0, 0, 0), 9)

                    used_el = num_current_el
                    result.append(data_stock_text[num_current_el])

                    # temp_img = img[y1:y1 + h1, x1:x1 + w1]
                    # res = calc_diff(temp_img, sess)
                    #
                    # print(res)

                    break

                # if тут мы будем проверять вот то, что написал на тетрадке, нужно проверку осуществить элемента, current_el
                # типо чтоб он, а точнее наши выдуманные координаты входили в какой-либо объект, если разность их, условно,
                # больше 100, то flag = False и весь код ретюрнется, что будет означать отсутвие 1. Элемента 2. Текста в целом

    print(result)

    # for i, el in enumerate(data.splitlines()):
    #     if i == 0:
    #         continue
    #     el = el.split()
    #     if len(el) > 11:
    #         print(el[11])
    #         for phrase in envelope_phrase: # нам же нужно чекать если есть это слово из словаря, то обводка делается, сейчас же оно абсолюьно любое слово походу трекает
    #             if phrase in el[11].lower():
    #                 try:
    #                     x, y, w, h = int(el[6]), int(el[7]), int(el[8]), int(el[9])
    #                     cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 1)
    #                     cv2.putText(img, el[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    #                 except IndexError:
    #                     print("я долбаеб")

    img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
    cv2.imshow('Result', img)
    cv2.waitKey(0)

    return result
