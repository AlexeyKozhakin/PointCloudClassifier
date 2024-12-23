import pandas as pd
import os

# Путь к каталогу с файлами .csv
input_dir = r'/mnt/working-ssd/alexey_kozhakin/MUSAC/data/STPLS3D/ml_data_training/cut_train_32_csv'  # Замените на путь к вашему каталогу
output_file = r'/mnt/working-ssd/alexey_kozhakin/MUSAC/data/STPLS3D/ml_data_training/dataset_train.csv'  # Замените на путь для итогового файла

# Список для хранения DataFrame
dataframes_list = []

# Чтение файлов .csv и объединение их в один DataFrame
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)
        dataframes_list.append(df)

# Проверка типов элементов в списке DataFrame
print(f"Типы элементов в списке: {type(dataframes_list[0])}")

# Объединение всех DataFrame в один
result_dataframe = pd.concat(dataframes_list, ignore_index=True)

# Сохранение результата в новый файл .csv
result_dataframe.to_csv(output_file, index=False)

print(f'Объединенный DataFrame сохранен в {output_file}')
