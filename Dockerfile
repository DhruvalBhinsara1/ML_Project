# Multi-Language Production Image: Python 3.11 + R 4.x
FROM python:3.11-slim-bookworm

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=5001

# Install R runtime and essential compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-install required R statistical & ML packages
RUN Rscript -e "install.packages(c('rpart', 'randomForest', 'dplyr', 'readr'), repos='https://cloud.r-project.org')"

# Set application working directory
WORKDIR /app

# Install Python production requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, static assets, and templates
COPY . .

# Ensure runtime directories exist with write permissions for generated plots (UID 1000 compatible for HF Spaces)
RUN mkdir -p static outputs plots models results data/raw && \
    chmod -R 777 static outputs plots models results

# Expose standard ports (7860 for Hugging Face Spaces, 5001 for local/other hosts)
EXPOSE 7860 5001

# Start the Flask production application via Gunicorn WSGI
CMD exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 2 --threads 4 --timeout 120 "app.app:app"