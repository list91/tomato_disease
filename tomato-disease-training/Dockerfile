FROM python:3.9

# Создаем пользователя
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Устанавливаем системные зависимости
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Рабочая директория
WORKDIR /app

# Копируем requirements
COPY --chown=user ./hf_requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Копируем код
COPY --chown=user train_hf.py .
COPY --chown=user app.py .

# Порт для Gradio
EXPOSE 7860

# Запускаем приложение
CMD ["python", "train_hf.py", "&&", "python", "app.py"]
