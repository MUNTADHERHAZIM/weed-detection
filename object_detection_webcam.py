import cv2  # Импорт библиотеки OpenCV для работы с изображениями
from ultralytics import YOLO  # Импорт модуля YOLO для обнаружения объектов на изображениях
import cvzone  # Импорт библиотеки cvzone для работы с графическими элементами
import pandas as pd  # Импорт библиотеки pandas для работы с данными в формате таблицы
import time  # Импорт модуля time для работы со временем

# Загрузка модели и имен классов
model = YOLO(r"C:\Users\Muntadher\Desktop\weeb (Обнаружение сорняков)\best.pt")  # Путь к модели YOLO
classNames = ['crop', 'weed']  # Имена классов объектов: 'crop' (урожай) и 'weed' (сорняк)

# Открытие веб-камеры
cap = cv2.VideoCapture(1)  # Использование веб-камеры (для другой камеры измените значение на 1, 2 и т.д.)

# Создание DataFrame для хранения обнаруженных объектов
df = pd.DataFrame(columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

# Инициализация переменных для вычисления FPS
prev_frame_time = 0
new_frame_time = 

while True:
    new_frame_time = time.time()  # Запись текущего времени

    # Захват кадра с веб-камеры
    success, img = cap.read()  # Чтение кадра

    if not success:
        print("Ошибка чтения кадра с веб-камеры.")
        break

    # Обнаружение объектов
    results = model(img, stream=True)  # Получение результатов обнаружения объектов на кадре

    for r in results:
        for box in r.boxes:
            # Извлечение координат, уверенности и класса объекта
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            conf = float(box.conf[0])  # Уверенность в обнаружении объекта
            cls = int(box.cls[0])  # Класс объекта

            try:
                # Отображение ограничивающего прямоугольника и надписи с помощью библиотеки cvzone
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))  # Отображение ограничивающего прямоугольника
                cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)  # Отображение надписи

                # Добавление обнаруженного объекта в DataFrame
                df = pd.concat([df, pd.DataFrame({'Class': classNames[cls], 'Confidence': conf, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}, index=[0])], ignore_index=True)
            except IndexError:
                print("Ошибка: индекс вне диапазона. Пропуск обнаружения.")

    # Вычисление и отображение FPS
    fps = 1 / (new_frame_time - prev_frame_time)  # Вычисление FPS
    prev_frame_time = new_frame_time  # Обновление времени предыдущего кадра
    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Отображение FPS

    # Отображение кадра
    cv2.imshow("Image", img)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Запись обнаруженных объектов в файл Excel
df.to_excel('detections_webcam.xlsx', index=False)

# Освобождение ресурсов
cap.release()  # Отключение веб-камеры
cv2.destroyAllWindows()  # Закрытие окон
