import numpy as np

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

    # 2. Вычисляем ковариационные матрицы
    cov_matrices = np.einsum('ijk,ilk->ijl', centered_neighbors, centered_neighbors) / k_neighbors  # (4096, 3, 3)

    # 3. Собственные значения и собственные векторы
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)  # (4096, 3), (4096, 3, 3)

    # 4. Извлекаем нормали и кривизну
    normals = eigenvectors[:, :, 0]  # Нормали: собственные векторы, соответствующие λ3 (4096, 3)
    curvatures = eigenvalues[:, 0] / np.sum(eigenvalues, axis=1)  # Кривизна: λ3 / (λ1 + λ2 + λ3) (4096,)
    print('p_n:', points_neighbors.shape)
    dz_min = points[:,2] - np.min(points_neighbors, axis=2)[:, 2]
    dz_max = points[:, 2] - np.max(points_neighbors, axis=2)[:, 2]

    return normals, curvatures, centered_neighbors, std, skewness, excess, dz_min, dz_max

def combine_features(normals, curvatures, skewness, excess, dz_max, dz_min, points):
    """
    Объединяет все вычисленные признаки в единую матрицу.

    :param points: cp.ndarray, форма (4096, 3) — координаты точек.
    :param normals: cp.ndarray, форма (4096, 3) — нормали.
    :param curvatures: cp.ndarray, форма (4096,) — кривизна.
    :param skewness: cp.ndarray, форма (4096,3) — коэффициент асимметрии.
    :param excess: cp.ndarray, форма (4096,3) — эксцесс.
    :param dxi_mean: cp.ndarray, форма (4096, 3) — разность точка - среднее по соседям.
    :return: cp.ndarray, форма (4096, p+1) — объединенные признаки.
    """
    # Убедитесь, что все массивы имеют одинаковое количество измерений
    print(
   f"Normals shape: {normals.shape}, Curvatures shape: {curvatures.shape}, "
   f"Skewness shape: {skewness.shape}, excess shape: {excess.shape},  "
   f"dz_max.shape {dz_max.shape}"
   f"dz_min.shape {dz_min.shape}, points[:,6] shape: {points[:, 6].shape}")

    features = np.hstack((normals, curvatures[:, None], skewness, excess, dz_max[:, None], dz_min[:, None], points[:,6:7]))
    return features


