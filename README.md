# ML_Project - Real Estate Analytics & House Price Prediction

A comprehensive machine learning system in **R** with an interactive **Flask** web application for real estate market analysis:
- **Price Valuation (Linear Regression)**: Predict continuous property market valuations based on size, BHK, age, and location.
- **Segment Classification (Decision Tree + Random Forest)**: Classify properties into market tiers (*Budget*, *Mid-Range*, *Premium*).
- **Property Recommendation (K-Nearest Neighbors)**: Recommend property tier with class probability distributions based on structural and neighborhood metrics.
- **Hierarchical Clustering (Dendrograms)**: Group similar properties using distance metrics and linkage methods.

---

## Project Structure

```
├── app/
│   └── app.py                          # Flask web server & prediction API endpoints
├── data/
│   ├── raw/
│   │   └── india_housing_prices.csv    # Primary dataset (250,000 properties)
│   ├── train_data.csv                  # Prepared training dataset
│   └── test_data.csv                   # Prepared testing dataset
├── ml/
│   └── clustering.R                    # Standalone hierarchical clustering script
├── models/                             # Serialized .rds models & city encodings
│   ├── city_mapping.rds
│   ├── dt_classifier.rds
│   ├── rf_classifier.rds
│   ├── knn_model.rds                   # Serialized KNN bundle (k=11, preProcess)
│   ├── linear_regression.rds
│   └── linearRegression.rds
├── outputs/                            # Generated standalone plots
│   ├── dendrogram.png
│   └── linearRegression_actual_vs_predicted.png
├── plots/                              # Generated evaluation plots
│   ├── decision_tree.png
│   └── rf_feature_importance.png
├── results/                            # Model predictions and performance metrics CSVs
├── R/                                  # Member standalone scripts
│   ├── 01_data_preparation.R           # Data cleaning & train/test split
│   ├── 02_decision_tree.R              # Decision tree model training & evaluation
│   ├── 03_random_forest.R              # Random forest model training & evaluation
│   ├── 04_model_evaluation.R           # Comprehensive model evaluation
│   └── 05_knn.R                        # KNN hyperparameter tuning & evaluation
├── SRC/                                # Flask app backend R scripts
│   ├── classification.R                # Classification training with city pricing score
│   ├── dendrogram.R                    # Clustering script
│   ├── linear_regression.R             # Regression training
│   ├── models/linearRegression.R       # Standalone linear regression pipeline
│   ├── predict_classification.R        # Classification inference CLI
│   ├── predict_dendrogram.R            # Dendrogram generation CLI
│   ├── predict_knn.R                   # KNN inference CLI
│   ├── predict_regression.R            # Regression inference CLI
│   └── preprocessing.R                 # Shared data loading & preprocessing
├── static/                             # Web assets & dynamically generated images
│   ├── css/style.css                   # Modular stylesheet
│   ├── decision_tree.png
│   ├── dendrogram.png
│   └── knn_accuracy_vs_k.png           # KNN optimization curve
├── templates/                          # Modular Jinja2 web UI templates
│   ├── base.html                       # Global layout shell & lightbox modal
│   ├── home.html                       # Landing overview & quick start cards
│   ├── regression.html                 # Price valuation page (Linear Regression)
│   ├── classification.html             # Market tier classifier page (DT + RF)
│   ├── clustering.html                 # Hierarchical clustering & dendrogram page
│   ├── recommendation.html             # Property recommendation page (KNN)
│   ├── about.html                      # Project documentation & architecture page
│   └── components/                     # Reusable Jinja2 UI components
│       ├── sidebar.html                # Navigation sidebar with active route detection
│       ├── mobile_header.html          # Responsive mobile header
│       ├── city_options.html           # 42-city dropdown options
│       └── lightbox.html               # Chart zoom modal
├── tests/                              # Automated test suite
│   └── test_app.py                     # Route & API integration tests
└── requirements.txt                    # Python dependencies
```

---

## Requirements & Installation

### 1. R Environment
Requires **R 4.x** with the following packages:

```r
install.packages(c("readr", "dplyr", "ggplot2", "rpart", "rpart.plot", "randomForest", "caret"))
```

### 2. Python Environment
Requires **Python 3.9+**:

```bash
# Optional: create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Launch the Flask Web Application

```bash
python app/app.py
```
*(Or `./venv/bin/python app/app.py`)*

Then open your browser and navigate to:
👉 **`http://127.0.0.1:5001`**

#### Available Web Routes:
| Route | Method | Description |
| :--- | :--- | :--- |
| **`/`** | GET | Overview & Quick Start dashboard |
| **`/regression`** | GET | Price Valuation using Linear Regression (OLS) |
| **`/classification`** | GET | Property Tier Classification (Decision Tree & Random Forest) |
| **`/clustering`** | GET | Hierarchical Clustering & Dendrogram Generator |
| **`/recommendation`** | GET | Property Tier Recommendation (K-Nearest Neighbors) |
| **`/about`** | GET | Project Documentation, Model Mathematics & Architecture |
| **`/predict_regression`** | POST | Computes continuous valuation (`size`, `bhk`, `age`, `city`) |
| **`/predict_classification`**| POST | Predicts tier using Decision Tree & Random Forest |
| **`/generate_dendrogram`** | POST | Generates hierarchical clustering dendrogram plot |
| **`/predict_knn`** | POST | Evaluates K-NN tier probabilities (`bhk`, `size`, `year_built`, etc.) |

> **Note**: The web app runs on port `5001` by default to avoid macOS AirPlay Receiver port 5000 conflicts. You can customize the port using the `PORT` environment variable:
> ```bash
> PORT=8080 python app/app.py
> ```

---

### 2. Run Standalone R Modeling Pipelines

* **Linear Regression Model & Visualizations:**
  ```bash
  Rscript SRC/models/linearRegression.R
  ```

* **Hierarchical Clustering & Dendrogram Plotting:**
  ```bash
  Rscript ml/clustering.R
  ```

* **Classification Pipeline (Data Preparation, Decision Tree, Random Forest, KNN):**
  ```bash
  Rscript R/01_data_preparation.R
  Rscript R/02_decision_tree.R
  Rscript R/03_random_forest.R
  Rscript R/04_model_evaluation.R
  Rscript R/05_knn.R
  ```

* **Train Web App Models (Fast City-Encoded Classification & Regression):**
  ```bash
  Rscript SRC/classification.R
  Rscript SRC/linear_regression.R
  ```

---

## Model Performance

| Model | Task | Metrics |
| :--- | :--- | :--- |
| **Linear Regression** | Price Prediction (Continuous) | $R^2 = 0.9643$, MAE: $9.96$ Lakhs |
| **Random Forest** | Tier Classification (Multiclass) | Accuracy: **$93.35\%$**, F1: **$0.933$** |
| **KNN (K=11)** | Tier Classification (Multiclass) | Accuracy: **$88.68\%$**, F1: **$0.887$** |
| **Decision Tree** | Tier Classification (Multiclass) | Accuracy: **$88.44\%$**, F1: **$0.883$** |
| **Hierarchical Clustering** | Property Grouping (Unsupervised) | Ward's method on Euclidean distance |

---

## 🚀 Production Deployment

The application is containerized with Docker to support its hybrid Python + R architecture and bundles all pretrained models in `models/`.

Detailed step-by-step instructions for popular platforms are available in [DEPLOYMENT.md](DEPLOYMENT.md):
- **Render** (Recommended — GitHub auto-deploy)
- **Railway** (One-click Docker deploy)
- **Google Cloud Run** (Serverless container with free tier)
- **Hugging Face Spaces** (Free 16GB Docker space)
- **Linux VPS / Docker Compose** (`docker compose up -d`)

Quick local Docker test:
```bash
docker compose up --build
```
