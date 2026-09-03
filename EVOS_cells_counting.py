import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import index_natsorted

# ==========================================
# Пути
# ==========================================

input_folder = input("Enter the images folder path: ")
output_csv = input("Enter the csv file name (without extension): ") + '.csv'

# ==========================================
# Параметры
# ==========================================

MIN_AREA = 20
MAX_AREA = 160

# ==========================================
# Функция подсчёта точек
# ==========================================


def count_spots(image_path: str, color_type):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # Переводим в HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ======================================
    # Зеленый цвет
    # ======================================

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # ======================================
    # Красный цвет
    # ======================================

    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    red_mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    red_mask = cv2.bitwise_or(
        red_mask1,
        red_mask2
    )

    # ======================================
    # Выбираем нужную маску
    # ======================================

    if color_type == "RFP":
        mask = red_mask

    elif color_type == "GFP":
        mask = green_mask

    elif color_type == "Live+Dead":
        mask = cv2.bitwise_or(
            red_mask,
            green_mask
        )

    else:
        return None

    # ======================================
    # Убираем шум
    # ======================================

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ======================================
    # Поиск объектов
    # ======================================

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(mask)

    count = 0

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if MIN_AREA <= area <= MAX_AREA:
            count += 1

    return count


# ==========================================
# Обработка изображений
# ==========================================

results = {}

for image_path in Path(input_folder).glob("*.*"):

    filename = image_path.name

    # ======================================
    # Определяем тип изображения
    # ======================================

    if "RFP" in filename:

        color_type = "RFP"

        # Убираем _RFP из названия
        base_name = filename.replace("_RFP", "")

    elif "GFP" in filename:

        color_type = "GFP"

        # Убираем _GFP из названия
        base_name = filename.replace("_GFP", "")

    else:

        color_type = "Live+Dead"

        # Название оставляем как есть
        base_name = filename

    # ======================================
    # Считаем точки
    # ======================================

    count = count_spots(
        image_path,
        color_type
    )

    if count is None:
        continue

    # ======================================
    # Создаём запись
    # ======================================

    if base_name not in results:

        results[base_name] = {
            "image": base_name,
            "RFP": "",
            "GFP": "",
            "Live+Dead": ""
        }

    results[base_name][color_type] = count


# ==========================================
# Создание DataFrame
# ==========================================

df = pd.DataFrame(results.values())

# ==========================================
# Натуральная сортировка
# ==========================================

df = df.iloc[
    index_natsorted(df["image"])
]

# ==========================================
# Сохранение CSV
# ==========================================

df.to_csv(
    output_csv,
    index=False
)

print(df)

print(f"\nCSV сохранён: {output_csv}")