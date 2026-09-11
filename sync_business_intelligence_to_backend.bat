@echo off
setlocal

cd /d C:\Users\User\Desktop\car-valuation-backend

echo Publishing OtoDeger Business intelligence files...
python sync_business_intelligence_to_backend.py

if errorlevel 1 (
    echo.
    echo ERROR: Business intelligence publish failed.
    exit /b 1
)

echo.
echo Business intelligence publish completed successfully.
exit /b 0
