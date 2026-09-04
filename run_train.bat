chcp 65001 >nul
@echo off
echo ========================================
echo MNIST 手写数字识别 - 训练脚本 (Windows)
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python，请先安装 Python
    pause
    exit /b 1
)

REM 检查并创建虚拟环境
if not exist "venv" (
    echo 📦 创建虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境
echo 🔄 激活虚拟环境...
call venv\Scripts\activate.bat

REM 安装依赖
echo 📦 安装依赖包...
pip install -r requirements.txt

REM 运行训练
echo.
echo 🚀 开始训练...
python train.py

echo.
echo ========================================
echo 训练完成！
echo ========================================
pause