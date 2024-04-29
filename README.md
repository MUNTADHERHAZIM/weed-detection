# Проект по обнаружению сорняков

## Обзор
Данный проект направлен на обнаружение и классификацию сельскохозяйственных культур и сорняков на изображениях и видео с использованием алгоритма детектирования объектов YOLO (You Only Look Once). Проект состоит из четырех основных частей:

1. Подготовка данных
2. Обучение первой модели
3. Обучение второй модели
4. Слияние моделей и оценка результатов

---

## Описание файлов

### 1. Подготовка данных
- **`data_preparation.ipynb`**: Этот ноутбук используется для скачивания набора данных и разделения его на две части: обучающую и валидационную.

### 2. Обучение первой модели
- **`train_model1.ipynb`**: Этот ноутбук используется для обучения первой модели YOLO с использованием обучающего набора данных, подготовленного в первой части.
- **`out_1.yaml`**: Файл конфигурации для первой модели
    ```yaml
    train: /content/output_1/train  # Путь к обучающему набору данных
    val: /content/output_1/val      # Путь к валидационному набору данных
    test: /content/output_1/test    # Путь к тестовому набору данных
    nc: 2                           # Количество классов объектов для обнаружения
    names: ['crop', 'weed']         # Наименования классов: 0 - crop (культура), 1 - weed (сорняк)
    ```

### 3. Обучение второй модели
- **`train_model2.ipynb`**: Этот ноутбук используется для обучения второй модели YOLO с использованием того же обучающего набора данных, подготовленного в первой части.
- **`out_2.yaml`**: Файл конфигурации для второй модели
    ```yaml
    train: /content/output_2/train  # Путь к обучающему набору данных
    val: /content/output_2/val      # Путь к валидационному набору данных
    test: /content/output_2/test    # Путь к тестовому набору данных
    nc: 2                           # Количество классов объектов для обнаружения
    names: ['crop', 'weed']         # Наименования классов: 0 - crop (культура), 1 - weed (сорняк)
    ```

### 4. Слияние моделей и оценка результатов
- **`model_fusion_and_evaluation.ipynb`**: Этот ноутбук используется для слияния двух обученных моделей YOLO и оценки производительности объединенной модели.

---

## Требования

- Python 3.x
- PyTorch
- OpenCV
- cvzone
- Pandas
- Ultralytics YOLO

---

## Установка

1. Клонируйте репозиторий:
    ```bash
    https://github.com/MUNTADHERHAZIM/weed-detection.git
    ```
2. Перейдите в директорию проекта:
    ```bash
    cd weed-detection
    ```

---

## Использование

1. **Подготовка данных**
    - Откройте и выполните ноутбук `data_preparation.ipynb` для скачивания набора данных и разделения его на обучающий и валидационный наборы.

2. **Обучение первой модели**
    - Откройте и выполните ноутбук `train_model1.ipynb` для обучения первой модели YOLO.

3. **Обучение второй модели**
    - Откройте и выполните ноутбук `train_model2.ipynb` для обучения второй модели YOLO.

4. **Слияние моделей и оценка результатов**
    - Откройте и выполните ноутбук `marge_models.ipynb` для слияния двух обученных моделей и оценки производительности объединенной модели.

---

## Результаты

Результаты объединенной модели сохранены в Excel-файле с именем `detections_combined.xlsx`.
и сохранены video с именем `output_video.avi`

result_model1
| epoch | train/box_loss | train/cls_loss | train/dfl_loss | metrics/precision(B) | metrics/recall(B) | metrics/mAP50(B) | metrics/mAP50-95(B) | val/box_loss | val/cls_loss | val/dfl_loss | lr/pg0 | lr/pg1 | lr/pg2 |
|-------|----------------|----------------|----------------|----------------------|-------------------|------------------|----------------------|--------------|--------------|---------------|--------|--------|--------|
| 1     | 1.4071         | 2.3943         | 1.6957         | 0.5369               | 0.30384           | 0.38391          | 0.19073              | 1.5308       | 2.9618       | 2.1361        | 0.00054592 | 0.00054592 | 0.00054592 |
| 2     | 1.3115         | 1.8022         | 1.6019         | 0.51583              | 0.45267           | 0.51893          | 0.25204              | 1.6596       | 3.496        | 2.222         | 0.00073806 | 0.00073806 | 0.00073806 |
| 3     | 1.3323         | 1.6312         | 1.62           | 0.76098              | 0.73315           | 0.76619          | 0.3978               | 1.6029       | 1.9991       | 2.1678        | 0.00056347 | 0.00056347 | 0.00056347 |


https://github.com/MUNTADHERHAZIM/weed-detection/assets/116030343/ac7599cc-0a51-48a3-9fa9-65af1a8bee9e

![1 (2)](https://github.com/MUNTADHERHAZIM/weed-detection/assets/116030343/20e2f0ed-085c-4399-bd0f-8201b95a6801)
![1 (1)](https://github.com/MUNTADHERHAZIM/weed-detection/assets/116030343/f1e5368f-d54f-4773-b62d-72bcf131969b)
![1 (1)](https://github.com/MUNTADHERHAZIM/weed-detection/assets/116030343/40137adf-ec99-4390-b6a8-97ffa49a22ea)

---


## Анализ результатов
**Значения потерь (Loss):**
- Средние значения потерь (box_loss, cls_loss, dfl_loss) на тренировочном наборе данных снижаются с каждой эпохой, что свидетельствует о хорошем обучении модели.
- Однако на валидационном наборе данных значения потерь немного колеблются. Это может указывать на нестабильность обучения или недостаточное количество данных для валидации.
**Метрики точности (Precision), полноты (Recall) и средней точности (mAP):**
Метрики точности и полноты (precision, recall) на тренировочном наборе данных в целом увеличиваются с увеличением числа эпох, что может говорить о том, что модель становится более точной и полной с каждой эпохой.
Метрика средней точности (mAP) также показывает положительную тенденцию увеличения на тренировочном наборе данных.
**Скорость обучения (Learning Rate):**
Скорость обучения (lr) остается примерно постоянной на протяжении всех эпох, что может указывать на эффективное управление процессом обучения и выбор оптимальной скорости обучения.

--- 

## Вывод
На основе анализа результатов обучения модели можно сделать следующие выводы:

- Обе модели (Model 1 и Model 2) демонстрируют улучшение показателей на тренировочном наборе данных с увеличением числа эпох, что указывает на эффективность обучения.
- Однако необходимо более внимательно рассмотреть изменения метрик и потерь на валидационном наборе данных для проверки стабильности обучения и избежания переобучения.
- При необходимости можно принять дополнительные меры, такие как изменение архитектуры модели, оптимизация параметров обучения или расширение тренировочного набора данных, для дальнейшего улучшения производительности модели.
