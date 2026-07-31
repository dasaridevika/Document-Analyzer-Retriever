# Dockerfile for Railway App Deployment (FastAPI Backend + Streamlit Frontend)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF & build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create volume storage directory
RUN mkdir -p /app/storage

# Copy application files
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/storage
ENV BACKEND_URL=http://127.0.0.1:8000

EXPOSE 8000 8501

# Run both FastAPI backend and Streamlit frontend concurrently
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
