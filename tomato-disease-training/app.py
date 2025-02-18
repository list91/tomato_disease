import os
import cv2
import numpy as np
import torch
import gradio as gr
from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Список классов болезней
DISEASE_CLASSES = [
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image(image):
    """Подготовка изображения для предсказания"""
    if image is None:
        return None
    
    # Resize и flatten
    img_resized = cv2.resize(image, (64, 64))
    img_flattened = img_resized.flatten()
    
    return img_flattened

def load_model():
    """Загрузка обученной модели"""
    try:
        # Загрузка модели SVM
        model_path = '/tmp/data/state/SVC_comb_R.pth.pth'
        
        # Если модель не существует, возвращаем None
        if not os.path.exists(model_path):
            print(f"Модель не найдена по пути: {model_path}")
            return None, None
        
        model_data = torch.load(model_path)
        
        # Создание pipeline с масштабированием
        scaler = StandardScaler()
        scaler.mean_ = model_data['mean']
        scaler.scale_ = model_data['std']
        
        classifier = model_data['classifier']
        
        return scaler, classifier
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None, None

def predict_disease(image):
    """Предсказание болезни томата"""
    if image is None:
        return "Пожалуйста, загрузите изображение"
    
    # Загрузка модели
    scaler, classifier = load_model()
    if scaler is None or classifier is None:
        return "Ошибка загрузки модели. Возможно, нужно сначала обучить модель."
    
    # Предобработка изображения
    processed_image = preprocess_image(image)
    if processed_image is None:
        return "Не удалось обработать изображение"
    
    # Масштабирование
    processed_image = scaler.transform([processed_image])
    
    # Предсказание
    prediction = classifier.predict(processed_image)
    probabilities = classifier.predict_proba(processed_image)[0]
    
    # Формирование результата
    result = f"Обнаружено: {prediction[0]}\n\n"
    result += "Вероятности:\n"
    for disease, prob in zip(DISEASE_CLASSES, probabilities):
        result += f"{disease}: {prob*100:.2f}%\n"
    
    return result

# FastAPI приложение
app = FastAPI()

# Gradio интерфейс
iface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="numpy", label="Загрузите изображение листа томата"),
    outputs=gr.Textbox(label="Результат диагностики"),
    title="Диагностика болезней томатов",
    description="Загрузите изображение листа томата для определения заболевания"
)

# Маршрут для Gradio
@app.get("/")
def read_root():
    return {"status": "Tomato Disease Classifier is running"}

# Запуск Gradio
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
