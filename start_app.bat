@echo off

REM Move to the folder where this file exists
cd /d %~dp0

REM Create virtual environment if it does not exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Start the Streamlit application
echo Starting CA Legal Assistant...
python -m streamlit run app.py

pause
