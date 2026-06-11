#!/bin/bash
# 设置 CUDA 环境并使用 conda 环境的 nvcc 编译 flash-attn

# 安装 CUDA 12.8 toolkit 到 conda 环境
# echo "Installing CUDA 12.8 toolkit..."
# conda install -c nvidia cuda-toolkit=12.8 cuda-nvcc=12.8 -y
# echo ""

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

echo "NVCC version:"
nvcc -V

echo ""
echo "Starting flash-attn compilation..."
MAX_JOBS=1 python -m pip install flash-attn==2.7.3 --no-build-isolation -v
