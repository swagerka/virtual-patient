import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv()

# Инициализация клиента OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Функция для генерации текста через OpenAI API
def generate_text(prompt):
    try:
        response = client.completions.create(
            model="text-davinci-003",  # Можно заменить на другую модель, например, "gpt-3.5-turbo"
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Ошибка: {str(e)}"

# Основной интерфейс Streamlit
st.title("Виртуальный Пациент")

# Пример сценария
st.subheader("Сценарий: Острый аппендицит")
st.write("**Пациент**: Мужчина, 32 года")
st.write("**Жалобы**: Острая боль в правой нижней части живота, тошнота, повышение температуры до 38°C")
st.write("**Задача**: Определите дальнейшие действия для диагностики и лечения.")

# Кнопка для генерации текста
if st.button("Сгенерировать описание симптомов"):
    prompt = "Опиши симптомы острого аппендицита у мужчины 32 лет."
    description = generate_text(prompt)
    st.write("**Сгенерированное описание**:")
    st.write(description)

# Запуск приложения
if __name__ == "__main__":
    st.write("Приложение запущено!")