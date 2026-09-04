#!/bin/bash
echo "========================================"
echo "MNIST 手写数字识别 - 训练脚本 (macOS/Linux)"
echo "========================================"
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3，请先安装 Python3"
    exit 1
fi

# 检查并创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "📦 安装依赖包..."
pip install -r requirements.txt

# 运行训练
echo ""
echo "🚀 开始训练..."
python3 src/train.py

echo ""
echo "========================================"
echo "训练完成!"
echo "========================================"