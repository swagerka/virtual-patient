@echo off
echo Activating virtual environment...

REM Проверяем, существует ли папка venv
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment 'venv' not found or activate.bat is missing.
    echo Please create the virtual environment first using: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Starting Streamlit application...
streamlit run app.py

echo Streamlit application has been closed or encountered an error.
pause