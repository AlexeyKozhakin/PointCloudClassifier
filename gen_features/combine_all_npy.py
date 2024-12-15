import numpy as np
import os

# Путь к каталогу с файлами .npz
input_dir = r'D:\data\las_org\test_parallel_preproc_32\test_npy_features'  # Замените на путь к вашему каталогу
output_file = r'D:\data\las_org\test_parallel_preproc_32\test_npy_features\dataset.npy'  # Замените на путь для итогового файла

# Список для хранения тензоров
tensor_list = []

# Чтение файлов .npz и объединение их в один массив
for filename in os.listdir(input_dir):
    if filename.endswith('.npy'):
        file_path = os.path.join(input_dir, filename)
        tensor = np.load(file_path)
        tensor_list.append(tensor)

print(tensor_list)

# Проверка типа элементов в списке тензоров
print(f"Типы элементов в списке: {type(tensor_list[0])}")

# Объединение всех тензоров в один массив (если массивы вложены в npz как отдельные массивы)
# Для этого можно использовать np.concatenate с параметром axis=0
result_tensor = np.concatenate(tensor_list, axis=0)

# Сохранение результата в новый файл .npy
np.save(output_file, result_tensor)

print(f'Объединенный тензор сохранен в {output_file}')
