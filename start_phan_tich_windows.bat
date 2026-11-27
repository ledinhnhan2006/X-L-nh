@echo off
setlocal
cd /d "%~dp0"
if not exist requirements-web.txt (
 echo Hay giai nen TOAN BO tep ZIP vao mot thu muc, roi chay lai tep nay.
 pause
 exit /b 1
)
if exist weights_segmentation.pt (
 set "YOLO_MODEL_PATH=%~dp0weights_segmentation.pt"
) else (
 echo Chua co checkpoint segmentation. Van co the phan tich anh va them vat thu cong.
)
if exist .venv\Scripts\python.exe (
 call start_app.bat
) else (
 call start_windows.bat
)
exit /b %errorlevel%
