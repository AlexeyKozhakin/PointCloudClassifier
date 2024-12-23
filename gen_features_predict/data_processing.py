import numpy as np


def fit_polynomial(tensor):
    """
    Подгонка полинома второго порядка ко всем облакам точек.

    :param tensor: np.ndarray, форма (4096, 3, 64) — тензор облаков точек.
    :return: параметры полинома (коэффициенты) для каждого облака.
    """
    # Извлечение x, y и z из тензора
    x = tensor[:, 0, :]  # (4096, 64)
    y = tensor[:, 1, :]  # (4096, 64)
    z = tensor[:, 2, :]  # (4096, 64)

    # Составление матрицы A для всех облаков
    A = np.empty((tensor.shape[0], 64, 6))  # (4096, 64, 6)
    A[:, :, 0] = x ** 2
    A[:, :, 1] = y ** 2
    A[:, :, 2] = x * y
    A[:, :, 3] = x
    A[:, :, 4] = y
    A[:, :, 5] = 1

    # Решение системы уравнений A * coeffs = b с использованием метода наименьших квадратов
    coeffs = np.linalg.lstsq(A.reshape(-1, 6), z.reshape(-1), rcond=None)[0]

    # Параметры полинома для каждого облака
    return coeffs.reshape(-1, 6)  # (4096, 6)


def normal_and_curvature(coeffs, points):
    """
    Вычисление нормали и кривизны для всех облаков в заданной точке.

    :param coeffs: коэффициенты полинома.
    :param points: np.ndarray, форма (N, 2) — координаты точек (x, y) для оценки.
    :return: нормали и кривизны.
    """
    x0 = points[:, 0]  # координаты x
    y0 = points[:, 1]  # координаты y

    a = coeffs[:, 0]
    b = coeffs[:, 1]
    c = coeffs[:, 2]
    d = coeffs[:, 3]
    e = coeffs[:, 4]

    # Частные производные
    dz_dx = 2 * a * x0 + c * y0 + d
    dz_dy = 2 * b * y0 + c * x0 + e

    # Нормаль
    normal_vectors = np.column_stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)))
    normal_vectors /= np.linalg.norm(normal_vectors, axis=1)[:, np.newaxis]  # Нормализация

    # Вторые производные для кривизны
    d2z_dx2 = 2 * a
    d2z_dy2 = 2 * b
    d2z_dxdy = c

    # Кривизна по формуле Гаусса
    curvature_values = (d2z_dx2 * d2z_dy2 - d2z_dxdy ** 2) / (1 + dz_dx ** 2 + dz_dy ** 2) ** (3 / 2)

    return normal_vectors, curvature_values

def compute_neighbors(points, k_neighbors):
    # Фильтрация только каналов 0:3
    points_filtered = points[:, :3]
    # 1. Вычисляем попарные расстояния только для 3 каналов
    distances = np.linalg.norm(points_filtered[:, None, :] - points_filtered[None, :, :], axis=-1)  # (4096, 4096, 3)
    # 2. Находим индексы k_neighbors ближайших точек
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k_neighbors+1]  # Исключаем саму точку
    # 3. Собираем ближайшие точки по индексам
    neighbors = points[nearest_indices]  # (4096, k_neighbors, 6)
    neighbors = np.moveaxis(neighbors, 1, -1)
    return neighbors


def compute_normals_and_curvatures_stat(points, points_neighbors, k_neighbors):
    """
    Вычисляет нормали и кривизну с использованием CuPy (GPU-ускорение).

    :param points_neighbors: cp.ndarray, форма (4096, 3, 64) — координаты соседей.
    :return: tuple:
        - normals: cp.ndarray, форма (4096, 3) — нормали.
        - curvatures: cp.ndarray, форма (4096,) — кривизна.
    """
    # 1. Центрируем соседей относительно центра масс
    centroid = np.mean(points_neighbors, axis=2, keepdims=True)  # (4096, 64, 3) ->
    print('centroid.shape:', centroid.shape)
    centered_neighbors = points_neighbors - centroid  # (4096, 3, 64)

    # 1. Коэффициент асимметрии (Skewness)
    skewness = np.mean(centered_neighbors**3, axis=2)  # (4096, 3)

    # 2. Коэффициент эксцесса (Excess Kurtosis)
    excess = np.mean(centered_neighbors**4, axis=2) - 3  # (4096, 3)

    # 3. Стандартное отклонение (Std)
    std = np.std(centered_neighbors, axis=2)  # (4096, 3)

    # # 2. Вычисляем ковариационные матрицы
    # cov_matrices = np.einsum('ijk,ilk->ijl', centered_neighbors, centered_neighbors) / k_neighbors  # (4096, 3, 3)
    #
    # # 3. Собственные значения и собственные векторы
    # eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)  # (4096, 3), (4096, 3, 3)
    #
    # # 4. Извлекаем нормали и кривизну
    # normals = eigenvectors[:, :, 0]  # Нормали: собственные векторы, соответствующие λ3 (4096, 3)
    # curvatures = eigenvalues[:, 0] / np.sum(eigenvalues, axis=1)  # Кривизна: λ3 / (λ1 + λ2 + λ3) (4096,)

    # Подгонка поверхности и получение параметров полинома для всех облаков
    coeffs = fit_polynomial(points_neighbors[:,:3,:])

    # Вычисление нормали и кривизны в заданной точке для всех облаков
    normals, curvatures = normal_and_curvature(coeffs, points[:,:3])
    print('normals.shape:', normals.shape)
    print('curvature.shape:', curvatures.shape)

    print('p_n:', points_neighbors.shape)
    dz_min = points[:,2] - np.min(points_neighbors, axis=2)[:, 2]
    dz_max = points[:, 2] - np.max(points_neighbors, axis=2)[:, 2]

    return normals, curvatures, centered_neighbors, std, skewness, excess, dz_min, dz_max

import numpy as np
import pandas as pd

def combine_features(normals, curvatures,
                                centered_neighbors, std, skewness, excess,
                                dz_max, dz_min, points):
    """
    Объединяет все вычисленные признаки в единую матрицу DataFrame с названиями столбцов.

    :param points: np.ndarray, форма (4096, 3) — координаты точек.
    :param normals: np.ndarray, форма (4096, 3) — нормали.
    :param curvatures: np.ndarray, форма (4096,) — кривизна.
    :param skewness: np.ndarray, форма (4096, 3) — коэффициент асимметрии.
    :param excess: np.ndarray, форма (4096, 3) — эксцесс.
    :param dz_max: np.ndarray, форма (4096,) — максимальная разность по z.
    :param dz_min: np.ndarray, форма (4096,) — минимальная разность по z.
    :return: pd.DataFrame — объединенные признаки с названиями столбцов.
    """

    # Проверка форм массивов
    print(
        f"Normals shape: {normals.shape}, Curvatures shape: {curvatures.shape}, "
        f"centered_neighbors shape: {centered_neighbors.shape}, std shape: {std.shape}, Skewness shape: {skewness.shape}, Excess shape: {excess.shape}, "
        f"dz_max shape: {dz_max.shape}, dz_min shape: {dz_min.shape}, "
        f"points shape: {points.shape}"
    )

    # Объединение массивов в одну матрицу
    features = np.hstack((
        points[:,:6],
        normals,
        curvatures[:, None],
        std,
        skewness,
        excess,
        dz_max[:, None],
        dz_min[:, None],
    ))

    # Создание DataFrame с названиями столбцов
    column_names = [
        'x', 'y', 'z', 'r', 'g', 'b',  # 6
        'normal_x', 'normal_y', 'normal_z', #3
        'curvature', #1
        'std_x', 'std_y', 'std_z', 'std_r', 'std_g', 'std_b', #6
        'skewness_x', 'skewness_y', 'skewness_z', 'skewness_r', 'skewness_g', 'skewness_b', #6
        'excess_x', 'excess_y', 'excess_z', 'excess_r', 'excess_g', 'excess_b', #6
        'dz_max', 'dz_min', #2
    ]

    # Создание DataFrame
    df_features = pd.DataFrame(features, columns=column_names)

    return df_features



