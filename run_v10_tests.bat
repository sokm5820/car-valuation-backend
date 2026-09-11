@echo off
setlocal
cd /d %~dp0
python -m py_compile app.py otodeger_v10_state.py otodeger_v10_agent.py build_business_activity_v10.py
if errorlevel 1 exit /b 1
python test_otodeger_v10_state.py
if errorlevel 1 exit /b 1
python test_v10_gold_standard.py
if errorlevel 1 exit /b 1
echo.
echo V10 checks passed.
