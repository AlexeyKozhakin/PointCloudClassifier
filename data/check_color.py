import laspy

# Открытие LAS-файла
filename = r"D:\data\las_org\data_las_stpls3d\all_org_las_rgb\RA_points.las"
las = laspy.read(filename)

# Проверяем наличие атрибутов цвета
has_color = hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue")
if has_color:
    # Вычисляем среднее значение по каждому цветовому каналу
    mean_red = las.red.mean()
    mean_green = las.green.mean()
    mean_blue = las.blue.mean()
    
    print("LAS файл содержит данные о цвете точек.")
    print(f"Среднее значение красного канала (Red): {mean_red}")
    print(f"Среднее значение зеленого канала (Green): {mean_green}")
    print(f"Среднее значение синего канала (Blue): {mean_blue}")
else:
    print("Данные о цвете отсутствуют.")

