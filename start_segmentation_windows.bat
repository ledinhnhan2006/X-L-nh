@echo off
setlocal
cd /d "%~dp0"
if not exist requirements-web.txt (
 echo Hay giai nen TOAN BO tep ZIP truoc khi chay ung dung.
 pause
 exit /b 1
)
if not exist weights_segmentation.pt (
 echo Chua co weights_segmentation.pt. Hay tai tep mo hinh segmentation vao thu muc nay.
 pause
 exit /b 1
)
set "YOLO_MODEL_PATH=%~dp0weights_segmentation.pt"
if exist .venv\Scripts\python.exe (
 call start_app.bat
) else (
 call start_windows.bat
)
exit /b %errorlevel%
