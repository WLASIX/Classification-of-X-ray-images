import joblib
import os
import numpy as np 
from skimage.io import imread
from random_forest import rf_train_model
from grid_search import gs_train_model

def classify_image():
    model = joblib.load('model.pkl')
    for image_file in os.listdir(test_image_path):
        image = imread(os.path.join(test_image_path, image_file))
        image = image.flatten()
        image = np.array([image])
        prediction = model.predict(image)

        print(f'Предсказанный класс для {image_file}: {classes[prediction[0]]}')

def main():
    while True:
        print("\nВыберите действие:")
        print("1. Обучить нейронную сеть (Random Forest)")
        print("2. Обучить нейронную сеть (Grid Search)")
        print(f"3. Использовать файл {model_name} для классификации изображения")
        choice = input("Введите номер вашего выбора: ")

        if choice == '1':
            os.system('cls||clear')
            rf_train_model(classes)
        if choice == '2':
            os.system('cls||clear')
            gs_train_model(classes)
        elif choice == '3':
            os.system('cls||clear')
            classify_image()     

if __name__ == '__main__':
    model_name = 'model.pkl'
    test_image_path = 'test'
    classes = ['closed fracture', 'crack', 'no fracture', 'open fracture']
    main()