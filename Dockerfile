# Optimized Dockerfile for Railway Deployment
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF & build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
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

# Run Uvicorn backend on port 8001 and Streamlit frontend on Railway's $PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 127.0.0.1 --port 8001 & streamlit run frontend/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
