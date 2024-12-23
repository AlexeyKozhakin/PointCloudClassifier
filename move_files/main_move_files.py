import os
import shutil
import json
import pandas as pd

# Загрузка конфигурации из файла config.json
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Создание каталога, если его нет
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Копирование файлов
def copy_files(file_list, source_dir, target_dir):
    for file_name in file_list:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            print(f"Файл {file_name} скопирован в {target_dir}")
        else:
            print(f"Файл {file_name} не найден в {source_dir}")

# Основная функция
def main():
    # Путь к файлу конфигурации
    config_path = "config.json"

    # Чтение конфигурации
    config = load_config(config_path)
    csv_path = config.get("csv_path")
    source_dir = config.get("source_dir")
    target_dir = config.get("target_dir")

    # Проверка, что все пути указаны
    if not all([csv_path, source_dir, target_dir]):
        print("Ошибка: Не все пути указаны в config.json")
        return

    # Убедиться, что целевой каталог существует
    ensure_directory_exists(target_dir)

    # Чтение CSV и извлечение списка файлов
    try:
        df = pd.read_csv(csv_path)
        if "File Names" not in df.columns:
            print("Ошибка: В CSV-файле нет столбца 'File Names'")
            return
        file_list = df["File Names"].dropna().tolist()
    except Exception as e:
        print(f"Ошибка при чтении CSV: {e}")
        return

    # Копирование файлов
    copy_files(file_list, source_dir, target_dir)

if __name__ == "__main__":
    main()

