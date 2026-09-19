# Deployment Guide: House Predictor ML Platform

This guide explains how to deploy the **House Predictor** web application and its machine learning models (*Linear Regression, Decision Tree, Random Forest, Hierarchical Clustering*) to production.

---

## 🏗️ Deployment Architecture

The application uses a **hybrid Python + R architecture**:
- **Web Layer**: Python Flask served via **Gunicorn WSGI** (multi-threaded production server).
- **Inference Engine**: Statistical and machine learning models executed via **R 4.x** (`Rscript`).
- **Pretrained Models**: Serialized binary `.rds` files bundled directly in the `models/` directory:
  - `models/linear_regression.rds` (~70 MB)
  - `models/dt_classifier.rds` (~33 KB)
  - `models/rf_classifier.rds` (~600 KB)
  - `models/knn_model.rds` (~2.1 MB)
  - `models/property_catalog.rds` (~3.7 MB)
  - `models/city_mapping.rds` (~800 B)

Because the project relies on both Python and R, **Docker containerization** is the recommended deployment method. All necessary Docker files (`Dockerfile`, `.dockerignore`, `docker-compose.yml`) are already included in the repository.

---

## 🚀 Option 1: Deploy to Render (Recommended — Easy & Free/Low Cost)

Render natively supports Docker-based deployments directly from your GitHub repository.

### Steps:
1. **Sign in to Render**:
   - Go to [render.com](https://render.com) and log in with your GitHub account.
2. **Create a New Web Service**:
   - Click **New +** in the top navigation and select **Web Service**.
3. **Connect Repository**:
   - Select your repository: `DhruvalBhinsara1/ML_Project`.
4. **Configure Settings**:
   - **Name**: `house-predictor` (or your preferred name)
   - **Region**: Choose the closest region (e.g. Singapore, Frankfurt, Oregon)
   - **Runtime**: Render will automatically detect the **`Dockerfile`**.
   - **Instance Type**: 
     - *Free* (512 MB RAM) or *Starter* (1 GB / 2 GB RAM recommended for smooth tree rendering).
5. **Deploy**:
   - Click **Create Web Service**.
   - Render will build the Docker container, install Python & R packages, and deploy your live HTTPS URL (e.g. `https://house-predictor.onrender.com`).

---

## ⚡ Option 2: Deploy to Railway (One-Click)

Railway automatically detects Dockerfiles and sets up routing with minimal configuration.

### Steps:
1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select `DhruvalBhinsara1/ML_Project`.
4. Railway will automatically detect the `Dockerfile` and start the build.
5. In your Railway service settings:
   - Go to **Settings** → **Networking** → Click **Generate Domain**.
   - Your app will be live at `https://house-predictor-production.up.railway.app`.

---

## ☁️ Option 3: Deploy to Google Cloud Run (Serverless & Free Tier)

Google Cloud Run runs containers serverless-ly, automatically scaling to zero when idle (extremely cost-effective).

### Prerequisites:
- [Google Cloud SDK (gcloud CLI)](https://cloud.google.com/sdk/docs/install) installed and authenticated.

### Deploy Command:
Run the following from the project root directory:

```bash
gcloud run deploy house-predictor \
  --source . \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 120s
```

Cloud Run will build the container on Google Cloud Build and output a live HTTPS URL.

---

## 🤗 Option 4: Deploy to Hugging Face Spaces (Free Docker Hosting)

Hugging Face Spaces provides **free 2 vCPU + 16 GB RAM** hosting for Docker containers.

### Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. **Space Name**: `house-predictor`
3. **License**: `mit` or `apache-2.0`
4. **Space SDK**: Select **Docker** (Blank).
5. Clone your HF space and push the project files, or sync from GitHub using GitHub Actions.
6. Hugging Face will automatically build the container and provide a public URL.

---

## 🖥️ Option 5: Deploy on a Linux VPS (Ubuntu / Debian, AWS EC2, DigitalOcean)

If you have your own cloud VM or server:

### Method A: Using Docker & Docker Compose (Fastest & Cleanest)

1. **Install Docker**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose-v2
   sudo systemctl enable --now docker
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/DhruvalBhinsara1/ML_Project.git
   cd ML_Project
   ```

3. **Start the Container**:
   ```bash
   sudo docker compose up -d --build
   ```

4. **Verify**:
   The app will be running at `http://your-server-ip:5001`.

---

### Method B: Native System Installation (Without Docker)

If you prefer installing Python and R directly onto an Ubuntu/Debian server:

1. **Install Python, R, and System Libraries**:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv r-base r-base-dev \
     build-essential libcurl4-openssl-dev libssl-dev libxml2-dev
   ```

2. **Install R Packages**:
   ```bash
   sudo Rscript -e "install.packages(c('rpart', 'randomForest', 'dplyr', 'readr', 'caret', 'jsonlite'), repos='https://cloud.r-project.org')"
   ```

3. **Set Up Python Virtual Environment**:
   ```bash
   python3 -m venv venv
   ./venv/bin/pip install -r requirements.txt
   ```

4. **Run with Gunicorn**:
   ```bash
   ./venv/bin/gunicorn --bind 0.0.0.0:5001 --workers 2 --threads 4 "app.app:app"
   ```

5. **(Optional) Configure Nginx Reverse Proxy & Free SSL with Certbot**:
   ```nginx
   # /etc/nginx/sites-available/house-predictor
   server {
       server_name yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:5001;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```
   Enable site and issue certificate:
   ```bash
   sudo ln -s /etc/nginx/sites-available/house-predictor /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl reload nginx
   sudo certbot --nginx -d yourdomain.com
   ```

---

## 🧪 Testing Locally Before Deployment

You can test the exact production Docker build on your local machine:

```bash
# Build the Docker image
docker build -t house-predictor .

# Run the container
docker run -p 5001:5001 -e PORT=5001 house-predictor
```

Open **`http://localhost:5001`** in your browser to verify that all models, predictions, and plots load correctly.