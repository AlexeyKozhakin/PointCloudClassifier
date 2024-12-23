import pandas as pd

def save_results(output_path, features):
    # Сохранение DataFrame в CSV файл
    features.to_csv(output_path, index=False)
