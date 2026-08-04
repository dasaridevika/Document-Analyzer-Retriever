# Optimized Dockerfile for Single-Service Railway Deployment (FastAPI + Streamlit)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF & build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create volume storage directory
RUN mkdir -p /app/storage

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/storage
ENV BACKEND_URL=http://127.0.0.1:8001
ENV BACKEND_PORT=8001

EXPOSE 8001 8501

# Entrypoint supervisor manages FastAPI on port 8001 and Streamlit on $PORT
CMD ["python", "entrypoint.py"]
