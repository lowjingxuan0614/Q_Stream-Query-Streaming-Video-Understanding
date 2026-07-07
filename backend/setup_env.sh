#!/bin/bash

set -e

echo "Setting Hugging Face mirror..."
export HF_ENDPOINT=https://hf-mirror.com

echo "Installing system dependencies..."
apt-get update
apt-get install -y \
  libgl1-mesa-glx \
  unzip \
  zip \
  wget \
  file \
  ffmpeg \
  curl \
  zstd

# requirements.txt = shopping list
# Tsinghua mirror = where pip downloads from
# pip install -r = install everything in the shopping list

echo "Installing Python dependencies..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing extra Python dependencies for 4.5 / 4.6 / VLM event description..."
pip install scikit-learn joblib requests -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing backend API dependencies..."
pip install fastapi uvicorn pydantic requests -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing SQLite auth dependencies..."
pip install sqlalchemy passlib[bcrypt] python-jose -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Checking PyTorch / OpenCLIP environment..."
python -c "import torch, cv2, open_clip, numpy; print('Environment OK'); print('CUDA available:', torch.cuda.is_available())"

echo "Checking sklearn / joblib / requests..."
python -c "import sklearn, joblib, requests; print('sklearn/joblib/requests OK')"

echo "Checking SQLite auth dependencies..."
python -c "import sqlalchemy, passlib, jose; print('SQLite auth dependencies OK')"

echo "Checking ffmpeg..."
ffmpeg -version | head -1

echo "Checking Ollama..."
if command -v ollama >/dev/null 2>&1; then
    ollama --version || true
else
    echo "Ollama not found."
    echo "If ollama-linux-amd64.tar.zst exists, install manually:"
    echo "zstd -d /root/autodl-tmp/ollama-linux-amd64.tar.zst -c | tar -xf - -C /usr/local"
fi

echo "Checking Ollama server..."
if curl -s http://localhost:11434/api/tags >/tmp/ollama_tags.json; then
    echo "Ollama server is running."
    cat /tmp/ollama_tags.json
    echo ""
else
    echo "Ollama server is not running."
    echo "Start it with:"
    echo "ollama serve > /tmp/ollama.log 2>&1 &"
fi

echo "Checking qwen2.5vl model..."
if command -v ollama >/dev/null 2>&1; then
    if ollama list | grep -q "qwen2.5vl"; then
        echo "qwen2.5vl model found."
    else
        echo "qwen2.5vl model not found."
        echo "When GPU is available, run:"
        echo "ollama pull qwen2.5vl:7b"
    fi
fi

echo "Setup finished."