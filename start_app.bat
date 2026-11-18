@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 echo Chua cai thu vien. Hay chay start_windows.bat truoc.
 pause
 exit /b 1
)
if not defined YOLO_MODEL_PATH if not exist weights.pt (
 echo Hay chep weights.pt da tai tu Roboflow vao thu muc nay.
 echo Ung dung se mo de huong dan, nhung chua cham duoc YOLO.
)
.venv\Scripts\python.exe web_app.py --open-browser
if errorlevel 1 pause
