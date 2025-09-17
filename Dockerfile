# Используем официальный Python образ
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y pv sudo git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY app.py .
COPY config.yaml .

# Создаем директорию для логов
RUN mkdir -p /app/logs

# Открываем порт 8000
EXPOSE 8000

# Устанавливаем переменную окружения для Gemini API
ENV GEMINI_API_KEY=""

# Команда для запуска приложения
CMD ["python", "app.py"]
