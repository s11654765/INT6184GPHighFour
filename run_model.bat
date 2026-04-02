@echo off
cd /d "%~dp0"
REM 使用 Python Launcher，避免命中 Microsoft Store 占位 python.exe（会导致无任何输出）
py -3 model.py
if errorlevel 1 pause
