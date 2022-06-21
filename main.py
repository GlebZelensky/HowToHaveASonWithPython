import json
import cv2
import face_recognition
import numpy as np
from tqdm import tqdm

# считываем данные из json файла
with open('result.json', 'r', encoding='utf-8') as json_file:
    messages = json.load(json_file)["messages"]

all_profiles = []

# заполняем массив анкет
for message in messages:

    # проверка, является ли сообщение анкетой
    is_profile = message["from"] == "Дайвинчик | Leomatchbot" and "photo" in message and message["text"] != ''

    # если это анкета, добавляем в массив
    if is_profile:
        # только если это не моя анкета)
        if "Павел" in message["text"]:
            continue

        data = {
            "photo": message["photo"],
            "description": message["text"],
            "label": 0
        }
        all_profiles.append(data)

    # проверка, является ли сообщение оценкой анкеты
    is_estimation = message["from"] == "Павел" and (message["text"] == "👎" or message["text"] == "❤️")

    # если это оценка, то проставляем её последней анкете
    if is_estimation:
        last_elem = len(all_profiles) - 1
        if message["text"] == "👎":
            all_profiles[last_elem]["label"] = 0
        else:
            all_profiles[last_elem]["label"] = 1

vectors = []

# проходим по всем профилям из массива
for profile in tqdm(all_profiles):

    age = 0

    # получаем возраст из описания профиля, если не можем его получить, то считаем его за 0
    if isinstance(profile["description"], str) and profile["description"].split()[1][:-1].isdigit():
        age = int(profile["description"].split()[1][:-1])

    # выбираем профили старше 18 и с оценкой "лайк"
    if profile["label"] == 1 and age >= 18:

        # считываем картинку профиля
        img = cv2.imread(profile["photo"])

        # пытаемся получить вектор лица
        try:
            vector = face_recognition.face_encodings(img)[0]
        except IndexError:
            continue

        # добавляем полученный вектор в массив векторов
        vectors.append(vector)


# сохраняем все полученные вектора в отдельный файл
vectors = np.array(vectors)
np.save("vectors", vectors)