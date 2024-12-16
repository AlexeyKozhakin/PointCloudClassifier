import os
import laspy
import multiprocessing
import csv
from collections import Counter

import numpy as np


def process_las_file(file_path):
    """
    Обрабатывает LAS-файл, подсчитывая процентное содержание классов.
    Возвращает имя файла и статистику по классам.
    """
    # Чтение LAS-файла
    with laspy.open(file_path) as lasfile:
        las = lasfile.read()
        classes = las.classification  # Извлекаем классификации точек
        print(np.unique(classes))
        total_points = len(classes)  # Общее количество точек

        if total_points == 0:
            return os.path.basename(file_path), {}

        # Подсчет количества точек каждого класса
        class_counts = Counter(classes)

        # Вычисление процентного содержания классов
        class_percentages = {
            class_id: (count / total_points) * 100
            for class_id, count in class_counts.items()
        }

        return os.path.basename(file_path), class_percentages



def process_all_files(input_dir, output_csv):
    """
    Обрабатывает все LAS-файлы в директории и сохраняет отчет в формате CSV.
    """
    # Список всех LAS-файлов в директории
    las_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.las')]
    print(las_files)
    # Создаем пул процессов
    with multiprocessing.Pool(processes = 7) as pool:
        # Обрабатываем файлы параллельно
        results = pool.map(process_las_file, las_files)

    # Формируем заголовки для CSV (определяем все уникальные классы)
    all_classes = set()
    for _, class_percentages in results:
        if class_percentages:
            all_classes.update(class_percentages.keys())
    all_classes = sorted(all_classes)  # Упорядочиваем классы

    # Записываем результаты в CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Записываем заголовок
        header = ['Filename'] + [f'Class_{cls}' for cls in all_classes]
        writer.writerow(header)

        # Записываем данные
        for filename, class_percentages in results:
            row = [filename]
            if class_percentages:
                row += [class_percentages.get(cls, 0) for cls in all_classes]
            else:
                row += ["Error"] * len(all_classes)
            writer.writerow(row)

if __name__ == "__main__":
    input_directory = r"D:\data\las_org\data_las_stpls3d\all_org_las"  # Укажите путь к каталогу с LAS-файлами
    output_file = "output_statistics.csv"  # Укажите путь для сохранения CSV

    process_all_files(input_directory, output_file)
    print(f"Отчет сохранен в {output_file}")
