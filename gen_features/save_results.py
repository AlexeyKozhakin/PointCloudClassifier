import numpy as np

def save_results(output_path, features):
    np.save(output_path, features=features)
