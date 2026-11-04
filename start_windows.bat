@echo off
cd /d "%~dp0"
echo Kiem tra Python 3.12...
py -3.12 --version
if errorlevel 1 goto fail
echo Dang chuan bi moi truong Python, vui long cho...
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements-web.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe web_app.py --open-browser
if errorlevel 1 goto fail
exit /b 0
:fail
 echo Khong chay duoc. Hay chup lai loi trong cua so nay.
 pause
 exit /b 1
