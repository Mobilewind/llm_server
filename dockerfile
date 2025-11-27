FROM python:3.9-slim
# Dockerfile
#FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用文件
COPY app.py .

# 創建必要的目錄
RUN mkdir -p models cache

# 暴露端口
EXPOSE 8000

# 啟動應用
CMD ["python", "app.py"]