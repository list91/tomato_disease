import os
import numpy as np
import torch
import cv2
import logging
import time
from tqdm import tqdm
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/tmp/train_model.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_dataset(max_images_per_class=200):
    """Загрузка и выборка датасета"""
    logger.info("Начало загрузки датасета")
    
    # URL Plant Village Dataset
    dataset_url = "https://data.mendeley.com/public-files/bulk/PlantVillage_Dataset.zip"
    
    try:
        # Загрузка архива
        response = requests.get(dataset_url)
        
        # Проверка успешности загрузки
        if response.status_code != 200:
            raise Exception("Не удалось загрузить датасет")
        
        # Распаковка в память
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Создаем директории
            os.makedirs('/tmp/data/raw/color', exist_ok=True)
            
            # Список болезней томатов
            tomato_diseases = [
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            
            # Словарь для подсчета изображений
            disease_image_count = {disease: 0 for disease in tomato_diseases}
            
            for file in tqdm(z.namelist(), desc="Извлечение файлов"):
                # Фильтруем только нужные изображения томатов
                matched_disease = next((disease for disease in tomato_diseases if disease in file), None)
                
                if matched_disease and file.endswith('.jpg'):
                    # Ограничиваем количество изображений для каждого класса
                    if disease_image_count[matched_disease] < max_images_per_class:
                        z.extract(file, f'/tmp/data/raw/color/{matched_disease}')
                        disease_image_count[matched_disease] += 1
        
        logger.info("Датасет успешно загружен и распакован")
        logger.info("Количество изображений по классам:")
        for disease, count in disease_image_count.items():
            logger.info(f"{disease}: {count}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке датасета: {e}")
        raise

def load_images_from_folder(folder, max_images=None):
    """Загрузка изображений из папки"""
    images = []
    labels = []
    logger.info(f"Загрузка изображений из папки: {folder}")
    
    file_list = os.listdir(folder)
    if max_images:
        file_list = file_list[:max_images]
    
    for filename in tqdm(file_list, desc=f"Обработка {os.path.basename(folder)}"):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize и flatten изображения
                    img_resized = cv2.resize(img, (64, 64))
                    images.append(img_resized.flatten())
                    labels.append(os.path.basename(folder))
            except Exception as e:
                logger.warning(f"Ошибка при обработке {img_path}: {e}")
    
    logger.info(f"Загружено изображений: {len(images)}")
    return images, labels

def prepare_dataset(dataset_path):
    """Подготовка датасета"""
    X = []
    y = []
    tomato_diseases = [
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    logger.info("Начало подготовки датасета")
    for disease in tqdm(tomato_diseases, desc="Обработка категорий"):
        folder_path = os.path.join(dataset_path, 'raw', 'color', disease)
        if os.path.exists(folder_path):
            images, labels = load_images_from_folder(folder_path)
            X.extend(images)
            y.extend(labels)
    
    logger.info(f"Всего изображений в датасете: {len(X)}")
    return np.array(X), np.array(y)

def train_model(X, y):
    """Обучение модели с кросс-валидацией"""
    logger.info("Начало подготовки и обучения модели")
    
    # Разделение на train и test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Размер тренировочной выборки: {len(X_train)}")
    logger.info(f"Размер тестовой выборки: {len(X_test)}")
    
    # Создание pipeline с масштабированием и SVM
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', probability=True)
    )
    
    # Кросс-валидация
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logger.info(f"Кросс-валидация: {cv_scores}")
    logger.info(f"Средняя точность кросс-валидации: {cv_scores.mean() * 100:.2f}%")
    
    # Обучение модели
    logger.info("Начало обучения модели...")
    start_time = time.time()
    model.fit(X_train, y_train)
    
    # Оценка точности
    accuracy = model.score(X_test, y_test)
    logger.info(f"Точность модели: {accuracy * 100:.2f}%")
    
    # Подробный отчет о классификации
    y_pred = model.predict(X_test)
    logger.info("\nОтчет о классификации:")
    logger.info(classification_report(y_test, y_pred))
    
    # Матрица ошибок
    logger.info("\nМатрица ошибок:")
    logger.info(str(confusion_matrix(y_test, y_pred)))
    
    end_time = time.time()
    logger.info(f"Время обучения: {end_time - start_time:.2f} секунд")
    
    return model

def save_model(model, path):
    """Сохранение модели"""
    logger.info(f"Сохранение модели в {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'classifier': model,
        'mean': model.named_steps['standardscaler'].mean_,
        'std': model.named_steps['standardscaler'].scale_
    }, path)
    logger.info("Модель успешно сохранена")

def main():
    """Основная функция обучения"""
    logger.info("Начало процесса обучения")
    
    # Загрузка датасета
    download_dataset()
    
    # Подготовка данных
    X, y = prepare_dataset('/tmp/data')
    
    # Обучение модели
    model = train_model(X, y)
    
    # Сохранение модели
    save_model(model, '/tmp/data/state/SVC_comb_R.pth.pth')
    
    logger.info("Процесс обучения завершен успешно!")

if __name__ == '__main__':
    main()
