#!/bin/bash
echo "========================================"
echo "MNIST 手写数字识别 - 预测脚本 (macOS/Linux)"
echo "========================================"
echo ""

if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行 ./run_train.sh"
    exit 1
fi

if [ ! -f "mnist_model.h5" ]; then
    echo "❌ 模型文件不存在，请先运行 ./run_train.sh 训练模型"
    exit 1
fi

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source venv/bin/activate

# 如果有参数则传递，否则显示帮助
if [ -z "$1" ]; then
    python3 predict_sequence_advanced.py
else
    python3 ppredict_sequence_advanced.py "$1"
fi

echo ""