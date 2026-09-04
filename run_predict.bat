chcp 65001 >nul
@echo off
echo ========================================
echo MNIST 手写数字识别 - 预测脚本 (Windows)
echo ========================================
echo.

if not exist "venv" (
    echo ❌ 虚拟环境不存在，请先运行 run_train.bat
    pause
    exit /b 1
)

if not exist "mnist_model.h5" (
    echo ❌ 模型文件不存在，请先运行 run_train.bat 训练模型
    pause
    exit /b 1
)

REM 激活虚拟环境
echo 🔄 激活虚拟环境...
call venv\Scripts\activate.bat

REM 如果有参数则传递，否则显示帮助
if "%1"=="" (
    python predict.py
) else (
    python predict.py %1
)

echo.
pause