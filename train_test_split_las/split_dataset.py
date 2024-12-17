import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# Функция для вычисления расстояния Хеллингера
def hellinger_distance(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)

# Чтение CSV-файла
def read_csv(file_path):
    data = pd.read_csv(file_path)
    file_names = data.iloc[:, 0].values  # Названия файлов (первый столбец)
    distributions = data.iloc[:, 1:].values  # Распределения классов (остальные столбцы)
    return file_names, distributions

# Разделение файлов на два набора в заданной пропорции
def split_files(file_names, distributions, ratio=0.5, max_iterations=1000):
    n_files = len(file_names)
    n_split = int(n_files * ratio)

    # Начальное случайное разбиение
    indices = np.arange(n_files)
    np.random.shuffle(indices)
    split_a_indices = indices[:n_split]
    split_b_indices = indices[n_split:]

    split_a = distributions[split_a_indices]
    split_b = distributions[split_b_indices]

    # Оптимизация разбиения
    for _ in range(max_iterations):
        # Вычисляем текущие средние распределения
        mean_a = np.mean(split_a, axis=0)
        mean_b = np.mean(split_b, axis=0)

        # Текущее расстояние Хеллингера
        current_distance = hellinger_distance(mean_a, mean_b)

        # Попробуем поменять местами элементы и проверим улучшение
        improved = False
        for i in split_a_indices:
            for j in split_b_indices:
                # Пробуем обменять i и j
                new_split_a_indices = np.copy(split_a_indices)
                new_split_b_indices = np.copy(split_b_indices)

                new_split_a_indices[np.where(split_a_indices == i)] = j
                new_split_b_indices[np.where(split_b_indices == j)] = i

                # Пересчитываем распределения
                new_split_a = distributions[new_split_a_indices]
                new_split_b = distributions[new_split_b_indices]

                # Новое расстояние Хеллингера
                new_mean_a = np.mean(new_split_a, axis=0)
                new_mean_b = np.mean(new_split_b, axis=0)
                new_distance = hellinger_distance(new_mean_a, new_mean_b)

                # Если улучшение найдено, применяем его
                if new_distance < current_distance:
                    split_a_indices = new_split_a_indices
                    split_b_indices = new_split_b_indices
                    split_a = new_split_a
                    split_b = new_split_b
                    current_distance = new_distance
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break  # Выходим, если улучшений больше нет

    return file_names[split_a_indices], file_names[split_b_indices], split_a, split_b

# Пример использования
file_path = "output_statistics.csv"
file_names, distributions = read_csv(file_path)
ratio = 0.2

split_a_files, split_b_files, split_a_distributions, split_b_distributions = split_files(file_names, distributions, ratio=ratio)

# Сохраняем результат с распределениями классов
split_a_df = pd.DataFrame(
    np.hstack([split_a_files.reshape(-1, 1), split_a_distributions]),
    columns=["File Names"] + [f"Class_{i}" for i in range(distributions.shape[1])]
)
split_b_df = pd.DataFrame(
    np.hstack([split_b_files.reshape(-1, 1), split_b_distributions]),
    columns=["File Names"] + [f"Class_{i}" for i in range(distributions.shape[1])]
)

# Добавляем суммарное распределение в конец каждого файла, нормализуем и округляем
total_a = np.sum(split_a_distributions, axis=0)
total_b = np.sum(split_b_distributions, axis=0)
print(total_a)
# Нормализация
normalized_total_a = (total_a / total_a.sum()*100).round(2)
normalized_total_b = (total_b / total_b.sum()*100).round(2)
print(normalized_total_b)
# Добавляем итоговые строки в DataFrame
split_a_df.loc[len(split_a_df)] = ["TOTAL"] + list(normalized_total_a)
split_b_df.loc[len(split_b_df)] = ["TOTAL"] + list(normalized_total_b)

# Сохраняем в файлы
split_a_df.to_csv("split_a.csv", index=False)
split_b_df.to_csv("split_b.csv", index=False)

