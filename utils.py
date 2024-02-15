from PIL import ImageOps
from PIL import Image
import numpy as np
import os
import random
import string
import shutil

# Создание новых папок для классифицированных изображений
def create_folders(output_folder, classes):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    for class_name in classes:
        class_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

# Добавление случайного шума к изображению
def add_noise(image):
    arr = np.array(image)
    noise = np.random.normal(0, 0.25, arr.shape)
    noisy_arr = arr + noise
    noisy_arr_clipped = np.clip(noisy_arr, 0, 255)  # обрезка значений за пределами диапазона [0, 255]
    return Image.fromarray(np.uint8(noisy_arr_clipped))

# Сдвиг изображения
def shift_image(image, x_shift, y_shift):
    return image.transform(image.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))

# Генерация случайной строки
def random_string(length=10):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def data_augmentation(input_folder, output_folder, size=(512, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for class_folder in os.listdir(input_folder):
        class_input_folder = os.path.join(input_folder, class_folder)
        class_output_folder = os.path.join(output_folder, class_folder)
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)
        for image_file in os.listdir(class_input_folder):
            image_path = os.path.join(class_input_folder, image_file)
            img = Image.open(image_path)
            img_resized = img.resize(size)
            img_gray = img_resized.convert('L')  # преобразование изображения в градации серого
            
            # Создание модифицированных вариантов изображения
            img_rotated = img_gray.rotate(random.randint(20, 65))  # поворот изображения
            img_flipped = ImageOps.flip(img_gray)  # отражение изображения по вертикали
            img_noisy = add_noise(img_gray)  # добавление шума к изображению
            
            # Генерация случайного имени файла
            random_name = random_string() + '.jpg'
            
            # Сохранение изображений
            output_path = os.path.join(class_output_folder, random_name)
            img_gray.save(output_path)
            img_rotated.save(output_path.replace('.jpg', '_rotated.jpg'))
            img_flipped.save(output_path.replace('.jpg', '_flipped.jpg'))
            #img_noisy.save(output_path.replace('.jpg', '_noisy.jpg'))