import cv2  # Импорт библиотеки OpenCV для работы с изображениями и видео
from ultralytics import YOLO  # Импорт YOLO модели для обнаружения объектов
import cvzone  # Импорт библиотеки cvzone для рисования прямоугольников и текста
import pandas as pd  # Импорт библиотеки pandas для работы с данными в виде таблицы
import time  # Импорт модуля time для работы с временем

# Загрузка модели и названий классов
model = YOLO(r"C:\Users\Muntadher\Desktop\weeb (Обнаружение сорняков)\merged.pt")  # Путь к модели YOLO
classNames = ['crop', 'weed']  # Названия классов объектов: ['сельскохозяйственная культура', 'сорняк']

# Открытие источника видео (указать путь к видео или использовать веб-камеру)
cap = cv2.VideoCapture(r"C:\Users\Muntadher\Desktop\weeb (Обнаружение сорняков)\pred_t.mp4")

# Создание DataFrame для хранения обнаруженных объектов
df = pd.DataFrame(columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

# Инициализация переменных для расчета FPS (количество кадров в секунду)
prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()  # Получение текущего времени

    # Захват кадра
    success, img = cap.read()

    if not success:
        print("Ошибка при считывании кадра из видеопотока.")
        break

    # Обнаружение объектов
    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            # Извлечение координат ограничивающего прямоугольника, уверенности и класса
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            try:
                # Обработка возможного выхода индекса класса за пределы диапазона
                if 0 <= cls < len(classNames):
                    # Рисование ограничивающего прямоугольника и надписи с помощью cvzone
                    cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                else:
                    print(f"Предупреждение: Индекс класса {cls} вышел за пределы диапазона. Используется метка по умолчанию.")
                    cvzone.putTextRect(img, "Неизвестный класс", (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Добавление обнаружения в DataFrame
                df = pd.concat([df, pd.DataFrame({'Class': classNames[cls], 'Confidence': conf, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}, index=[0])], ignore_index=True)
            except IndexError:
                print("Ошибка: Индекс за пределами диапазона. Пропуск обнаружения.")

    # Расчет и отображение FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Показ кадра
    cv2.imshow("Изображение", img)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Запись обнаруженных объектов в файл Excel
df.to_excel('detections3.xlsx', index=False)

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()