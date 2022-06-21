import cv2
import pytesseract
import numpy as np
import os

from PostalService.src.recognizeHandwriteText import *
from PostalService.src.text_classificator import text_type_classification

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

envelope_phrase = ['ФИО', 'Откуда', 'Отправитель', 'Адрес', 'Куда', 'Кому', 'От кого', "Абонент", "Лицевой счет",
                   "Плательщик"]


def recognize_all_types_of_written_text(img_file, model):
    num_of_valid_data = {}

    result = []

    # windows
    pytesseract.pytesseract.tesseract_cmd = "../src/Tesseract-OCR/tesseract.exe"

    arr = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    # Remove shadow
    rgb_planes = cv2.split(image)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    img_without_shadow = cv2.merge(result_norm_planes)

    img_without_shadow = cv2.resize(img_without_shadow, (0, 0), fx=6, fy=6)

    ret, thresh1 = cv2.threshold(img_without_shadow, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    rects = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rects.append([x, y, w, h])

    img = img_without_shadow.copy()

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

        flag = False
        for rect in rects:
            if rect[0] <= data['left'][num[1] + 1] + data['width'][num[1] + 1] <= rect[0] + rect[2] and \
                    rect[1] <= data['top'][num[1] + 1] <= rect[1] + rect[3]:
                if rect[2] - rect[0] > data['width'][0]/1.2:
                    break

                crop_img2 = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                crop_img2 = cv2.resize(crop_img2, (0, 0), fx=1 / 6, fy=1 / 6)

                chance_type_classification = text_type_classification(crop_img2)

                print(chance_type_classification)

                bigger_chance_type_classification = list(chance_type_classification.items())[0][0]
                if bigger_chance_type_classification == "handwritten":

                    recognized, recognized_corrected = bot_run_model(model, crop_img2)
                    print(recognized, recognized_corrected)

                    result.append(recognized_corrected)
                    flag = True

        if flag:
            continue

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
