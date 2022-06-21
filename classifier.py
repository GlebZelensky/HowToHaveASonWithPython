import numpy as np
import face_recognition
import cv2
from scipy.spatial.distance import pdist


# считываем фото которое необходимо проверить
img = cv2.imread("micasa.jpg")

# пытаемся построить вектор лица
try:
    vector = face_recognition.face_encodings(img)[0]
except IndexError:
    print("Лицо не распознано!")
    exit(0)


# загружаем заранее сохранённый набор векторов
vectors = np.load("vectors.npy")
result = []


# проходим по набору векторов и сравниваем новое фото с каждым вектором
for elem in vectors:

    answer = pdist([vector, elem], "euclidean")
    result.append(answer[0])

# вывод результатов
print(f"Average from numpy: {np.average(result)}")
print(f"Median from numpy: {np.median(result)}")

# выводим финальный вердикт
if np.average(result) > 0.72:
    print("dislike")
else:
    print("like")
