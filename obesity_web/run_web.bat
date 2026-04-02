@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "xgb_model_bundle.joblib" (
  echo 未找到模型，正在训练...
  py -3 train_xgb_model.py
  if errorlevel 1 exit /b 1
)

echo 启动服务: http://127.0.0.1:5000
py -3 app.py
