import json
import laspy
import numpy as np
import cupy as cp
from multiprocessing import Pool


def load_config(config_path):
    """
    Загрузка конфигурации из JSON файла.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def process_las_file(file_path, max_points_per_tile=4096, tile_size=32):
    """
    Загрузка и обработка .las файла.
    """
    # Загрузка файла
    with laspy.read(file_path) as las:
        x = las.x
        y = las.y
        z = las.z
        r = getattr(las, 'red', None)  # Некоторые .las могут не иметь r, g, b
        g = getattr(las, 'green', None)
        b = getattr(las, 'blue', None)
        classes = las.classification

    # Создание массива точек
    points = np.vstack((x, y, z, r, g, b, classes)).T

    # Разделение на тайлы
    tiles = []
    for i in range(0, len(points), tile_size ** 2):
        tile = points[i:i + tile_size ** 2]
        if len(tile) >= max_points_per_tile:
            sampled_tile = tile[np.random.choice(len(tile), max_points_per_tile, replace=False)]
            tiles.append(sampled_tile)

    return tiles


def process_multiple_files(file_paths, max_points_per_tile=4096, tile_size=32, num_processes=4):
    """
    Параллельная обработка списка .las файлов.
    """
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_las_file, [(path, max_points_per_tile, tile_size) for path in file_paths])
    return [tile for result in results for tile in result]


# Загрузка конфигурации
config = load_config("config.json")

# Параметры из конфигурации
tile_size = config["tile_size"]
max_points_per_tile = config["max_points_per_tile"]
num_processes = config["num_processes"]
file_paths = config["input_files"]

# Обработка файлов
tiles = process_multiple_files(file_paths, max_points_per_tile=max_points_per_tile, tile_size=tile_size,
                               num_processes=num_processes)
print(f"Количество обработанных тайлов: {len(tiles)}")
