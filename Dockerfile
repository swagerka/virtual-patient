# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt requirements.txt

# Устанавливаем зависимости
# --no-cache-dir чтобы не сохранять кэш pip и уменьшить размер образа
# --default-timeout=100 чтобы увеличить таймаут для pip install, если интернет медленный
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Копируем все файлы приложения в рабочую директорию
COPY . .

# Указываем порт, который будет слушать Streamlit
EXPOSE 8501

# Переменная окружения для URL API (может быть переопределена при запуске контейнера)
ENV KOBOLD_API_URL="http://localhost:5002/v1/"

# Команда для запуска приложения Streamlit
# --server.headless true рекомендуется для запуска в контейнерах
# --server.enableCORS false может быть полезно в некоторых окружениях, но обычно не требуется
# --server.address 0.0.0.0 чтобы приложение было доступно извне контейнера
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]