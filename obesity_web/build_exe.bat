@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "xgb_model_bundle.joblib" (
  echo 请先运行 train_xgb_model.py 生成模型
  py -3 train_xgb_model.py
  if errorlevel 1 exit /b 1
)

echo 安装 PyInstaller（若尚未安装）...
py -3 -m pip install -q pyinstaller

echo 正在打包单文件 exe，可能需要几分钟...
py -3 -m PyInstaller obesity_predictor.spec
if errorlevel 1 exit /b 1

echo.
echo 完成：可执行文件位于 dist\ObesityPredictor.exe
echo 把该 exe 发给他人，双击即可（无需安装 Python）。
pause
