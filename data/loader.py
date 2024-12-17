import os
import json
import laspy
import numpy as np
import logging
from multiprocessing import Pool

def load_config(config_path):
    """
    Загрузка конфигурации из JSON файла.
    """
    with open(config_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    return config

def setup_logging(log_file):
    """
    Настройка логгирования.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8"
    )

def process_las_file(file_path, output_dir, max_points_per_tile=4096, tile_size=32):
    """
    Загрузка и обработка одного .las файла.
    Сохраняет результат в .npy файл.
    """
    try:
        # Чтение LAS файла
        with laspy.open(file_path) as lasfile:
            las = lasfile.read()
            x = las.x
            y = las.y
            z = las.z
            r = las.red  # Некоторые файлы могут не иметь r, g, b
            g = las.green
            b = las.blue
            classes = las.classification


        # Объединение точек в массив
        points = np.vstack((x, y, z, r, g, b, classes)).T
        # Разделение на тайлы
        if points.shape[0] >= max_points_per_tile:
            sampled_tile = points[np.random.choice(points.shape[0], max_points_per_tile, replace=False)]
        else:
            logging.warning(f"Файл {file_path} пропущен: недостаточно точек.")
            return

        output_file = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.las', f'.npy')}")
        np.save(output_file, sampled_tile)

        logging.info(f"Файл {file_path} успешно обработан")
    except Exception as e:
        logging.error(f"Ошибка при обработке {file_path}: {str(e)}")

def process_multiple_files(input_dir, output_dir, max_points_per_tile, tile_size, num_processes):
    """
    Параллельная обработка всех .las файлов в указанной директории.
    """
    # Получение списка файлов .las
    file_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".las")]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Параллельная обработка
    with Pool(processes=num_processes) as pool:
        pool.starmap(
            process_las_file,
            [(file_path, output_dir, max_points_per_tile, tile_size) for file_path in file_paths]
        )

# Основной код
if __name__ == "__main__":
    # Загрузка конфигурации
    config_path = "config.json"
    config = load_config(config_path)

    # Настройка параметров
    input_dir = config.get("input_directory", "input")
    output_dir = config.get("output_directory", "output")
    max_points_per_tile = config.get("max_points_per_tile", 4096)
    tile_size = config.get("tile_size", 32)
    num_processes = config.get("num_processes", 4)
    log_file = config.get("log_file", "processing.log")

    # Настройка логгирования
    setup_logging(log_file)

    # Начало обработки
    logging.info("Начало обработки файлов .las")
    process_multiple_files(input_dir, output_dir, max_points_per_tile, tile_size, num_processes)
    logging.info("Обработка завершена.")
