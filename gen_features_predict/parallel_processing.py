from multiprocessing import Pool
from functools import partial
from data_processing import compute_neighbors, compute_normals_and_curvatures_stat, combine_features
import numpy as np
from save_results import save_results
from config_loader import load_config
import os
from tqdm import tqdm

def process_file(file_path, output_dir, config):
    """
    Обрабатывает один файл: загружает данные, вычисляет признаки.
    """
    data = np.load(file_path)
    print(data.shape)
    points = data[:,:6]
    k_neighbors = config['k_neighbors']
    points_neighbors = compute_neighbors(points, k_neighbors)
    (normals, curvatures,
     centered_neighbors, std, skewness, excess,
     dz_min, dz_max) = compute_normals_and_curvatures_stat(points, points_neighbors, k_neighbors)
    features = combine_features(normals, curvatures,
                                centered_neighbors, std, skewness, excess,
                                dz_max, dz_min, data)
    output_file = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.npy', f'.csv')}")
    print('out_file:', output_file)
    print('features:', features.shape)
    save_results(output_file, features)


def process_multiple_files(input_dir, output_dir, config, num_processes):
    """
    Параллельная обработка множества файлов.
    """
    # Получаем список всех `.npy` файлов
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')]

    # Обернем process_file с config с помощью partial
    partial_process_file = partial(process_file, output_dir=output_dir, config=config)

    with Pool(num_processes) as pool:
        # tqdm для отображения прогресса
        list(tqdm(pool.imap_unordered(partial_process_file, files), total=len(files)))




def main(input_dir, output_dir, config):
    """
    Главная функция для запуска обработки.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_multiple_files(input_dir, output_dir, config, config['num_processes'])

if __name__ == "__main__":
    # Загрузка конфигурации
    config = load_config('config.json')
    main(config['input_directory'], config['output_directory'], config)
