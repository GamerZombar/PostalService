from pathlib import Path
import cv2
import pytesseract
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

envelope_phrase = ['ФИО', 'Откуда', 'Отправитель', 'Адрес', 'Куда', 'Кому', 'От кого', "Абонент", "Лицевой счет",
                   "Плательщик"]


def recognize_typewritten_text(img_file):
    num_of_valid_data = {}

    result = []

    # windows
    pytesseract.pytesseract.tesseract_cmd = str(Path(__file__).resolve().parent) + "\\Tesseract-OCR\\tesseract.exe"

    arr = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (0, 0), fx=6, fy=6)

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = thresh

    config = '--oem 3 --psm 6'

    data = pytesseract.image_to_data(img, lang="rus", config=config,
                                     output_type=pytesseract.Output.DICT)

    data_stock_text = data['text'].copy()

    n_boxes = len(data['level'])

    for i in range(n_boxes):
        data['text'][i] = data['text'][i].lower()
        for phrase in envelope_phrase:
            phrase_lower = phrase.lower()
            if phrase_lower in data['text'][i]:
                num_of_valid_data[phrase] = i
                break

            if phrase_lower.split()[0] in data['text'][i]:
                if phrase_lower.split()[1] in data['text'][i + 1]:
                    num_of_valid_data[phrase] = i + 1

    for num in num_of_valid_data.items():

        if num is None:
            break

        y = data['top'][num[1]]

        num_current_el = num[1]

        used_el = num[1]

        if not result:
            result.append(num[0] + ": ")
        else:
            result.append('\n' + num[0] + ": ")

        for j in range(n_boxes - num[1] - 2):
            num_current_el += 1

            for i in range(data['left'][num_current_el] + data['width'][num_current_el], data['width'][0], 10):
                current_el_x = data['left'][num_current_el]

                if num_current_el - 1 != num[1]:
                    if current_el_x - (data['left'][used_el] + data['width'][used_el]) >= 200:
                        break

                if data['top'][num_current_el] - 120 <= y <= data['top'][num_current_el] + 120:
                    used_el = num_current_el
                    result.append(data_stock_text[num_current_el])
                    break

    return result
