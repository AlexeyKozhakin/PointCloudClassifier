import laspy

# Открытие LAS-файла
filename = r"/home/alexey/MUSAC/data/Malta/san_gwann/453632_3974144.las"
las = laspy.read(filename)

    # Чтение основных метаданных
print("Point Format:", las.header.point_format)
print("Number of Points:", las.header.point_count)
print("Scale Factors:", las.header.scales)
print("Offsets:", las.header.offsets)
print("Bounds (X, Y, Z):", las.header.mins, "-", las.header.maxs)
    
    # Проверка наличия цветовых данных
print("Color Data is Present")
print("Red Min-Max:", las['red'].min(), "-", las['red'].max())
print("Green Min-Max:", las['green'].min(), "-", las['green'].max())
print("Blue Min-Max:", las['blue'].min(), "-", las['blue'].max())


print('las.red=',las.red)
print('las.green=',las.green)
print('las.blue=',las.blue)
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

