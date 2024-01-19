import os
import numpy as np
import shutil
import joblib
from utils import data_augmentation
from utils import create_folders
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Загрузка данных и меток
def load_data(folder_path, classes):
    images = []
    labels = []
    image_files = []
    for class_index, class_name in enumerate(classes):
        class_folder = os.path.join(folder_path, class_name)
        for image_file in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_file)
            #print(f'Файл - {image_path}')
            image = imread(image_path)
            images.append(image.flatten())
            labels.append(class_index)
            image_files.append(image_file) 
    return np.array(images), np.array(labels), image_files

# Классификация и перемещение изображений
def classify_and_move_images(model, X_test, y_test, test_folder, output_folder, classes, image_files):
    y_pred = model.predict(X_test)
    for i, (pred, true) in enumerate(zip(y_pred, y_test)):
        predicted_folder = os.path.join(output_folder, classes[pred])
        true_folder = os.path.join(test_folder, classes[true])
        image_file = image_files[i] 
        image_path = os.path.join(true_folder, image_file)
        shutil.copy(image_path, predicted_folder)

def gs_train_model(_classes):
    dataset_folder = 'resized_dataset'
    output_folder = 'classified_images'
    classes = _classes
    
    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)

    # Увеличение количества изображений и приведение их к одному размеру
    print('Аугментация данных...')
    data_augmentation('raw_dataset', dataset_folder)

    # Загрузка данных
    print('Загрузка данных...')
    X, y, image_files = load_data(dataset_folder, classes) 
    
    # Разделение данных на обучающую и тестовую выборки
    print('Разделение данных...')
    X_train, X_test, y_train, y_test, image_files_train, image_files_test = train_test_split(X, y, image_files, test_size = 0.2, random_state = 42)
    
    # Обучение модели
    print('Обучение модели...')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    model = RandomForestClassifier(random_state = 42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv = 5, verbose = 2)
    grid_search.fit(X_train, y_train)

    # Выбор лучшей модели
    model = grid_search.best_estimator_

    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division = 1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division = 1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division = 1)

    print(f'Точность (accuracy): {accuracy:.2f}')
    print(f'Точность (precision): {precision:.2f}')
    print(f'Полнота (recall): {recall:.2f}')
    print(f'F1-мера (f1-score): {f1:.2f}')
    
    # Сохранение модели
    joblib.dump(model, 'model.pkl')

    # Создание папок для классифицированных изображений
    create_folders(output_folder, classes)
    
    # Классификация и перемещение изображений
    classify_and_move_images(model, X_test, y_test, dataset_folder, output_folder, classes, image_files_test)