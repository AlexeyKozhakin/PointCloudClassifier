import pandas as pd
import laspy
import os

# Словарь с цветами для классов
class_colors_stpls3d = {
    0: (0, 0, 0),          # Ground - Черный
    1: (255, 255, 255),    # Building - Зеленый           1==17
    2: (255, 255, 0),      # LowVegetation - Желтый
    3: (0, 0, 255),        # MediumVegetation - Синий
    4: (255, 0, 0),        # HighVegetation - Красный
    5: (0, 255, 255),      # Vehicle - Бирюзовый
    6: (255, 0, 255),      # Truck - Магента
    7: (255, 128, 0),      # Aircraft - Оранжевый
    8: (255, 0, 255),      # MilitaryVehicle - Серый             8==6
    9: (255, 20, 147),     # Bike - Deep Pink
    10: (255, 69, 0),      # Motorcycle - Красный Апельсин
    11: (210, 180, 140),   # LightPole - Бежевый
    12: (255, 105, 180),   # StreetSign - Hot Pink
    13: (165, 42, 42),     # Clutter - Коричневый
    14: (139, 69, 19),     # Fence - Темно-коричневый
    15: (128, 0, 128),     # Road - Фиолетовый
    17: (128, 128, 128),   # Windows - Белый
    18: (222, 184, 135),   # Dirt - Седой
    19: (127, 255, 0),     # Grass - Ярко-зеленый
}

# Путь к каталогу с CSV файлами
input_directory = "/home/alexey/MUSAC/data/Malta/san_gwann/cut_predict_32_csv_class"
output_directory = "/home/alexey/MUSAC/data/Malta/san_gwann/cut_predict_32_las"
os.makedirs(output_directory, exist_ok=True)

# Обработка всех CSV файлов в каталоге
for csv_file in os.listdir(input_directory):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(input_directory, csv_file)
        
        # Чтение CSV файла
        df = pd.read_csv(csv_path)

        # Проверка наличия необходимых колонок
        if not {'x', 'y', 'z', 'class'}.issubset(df.columns):
            print(f"Пропущены необходимые колонки в файле: {csv_file}")
            continue

        # Добавление колонок r, g, b на основе класса
        df['r'], df['g'], df['b'] = zip(*df['class'].map(class_colors_stpls3d))

        # Создание LAS файла
        las_file_path = os.path.join(output_directory, os.path.splitext(csv_file)[0] + ".las")
        # Создание LAS файла с версией 1.2 и форматом точек 3
        las = laspy.create(file_version="1.2", point_format=3)


        las.x = df['x'].values
        las.y = df['y'].values
        las.z = df['z'].values
        las.classification = df['class'].values.astype('uint8')
        las.red = df['r'].values.astype('uint16')
        las.green = df['g'].values.astype('uint16')
        las.blue = df['b'].values.astype('uint16')
        # Сохранение LAS файла
        las.write(las_file_path)

        print(f"LAS файл создан: {las_file_path}")

print("Все файлы обработаны.")

