# ML_Project - Real Estate Analytics & House Price Prediction

A comprehensive machine learning system in **R** with an interactive **Flask** web application for real estate market analysis:
- **Price Valuation (Linear Regression)**: Predict continuous property market valuations based on size, BHK, age, and location.
- **Segment Classification (Decision Tree + Random Forest)**: Classify properties into market tiers (*Budget*, *Mid-Range*, *Premium*).
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
│   └── 04_model_evaluation.R           # Comprehensive model evaluation
├── SRC/                                # Flask app backend R scripts
│   ├── classification.R                # Classification training with city pricing score
│   ├── dendrogram.R                    # Clustering script
│   ├── linear_regression.R             # Regression training
│   ├── models/linearRegression.R       # Standalone linear regression pipeline
│   ├── predict_classification.R        # Classification inference CLI
│   ├── predict_dendrogram.R            # Dendrogram generation CLI
│   ├── predict_regression.R            # Regression inference CLI
│   └── preprocessing.R                 # Shared data loading & preprocessing
├── static/                             # Web assets & dynamically generated images
│   ├── css/style.css                   # Modular stylesheet
│   ├── decision_tree.png
│   └── dendrogram.png
├── templates/                          # Modular Jinja2 web UI templates
│   ├── base.html                       # Global layout shell & lightbox modal
│   ├── home.html                       # Landing overview & quick start cards
│   ├── regression.html                 # Price valuation page (Linear Regression)
│   ├── classification.html             # Market tier classifier page (DT + RF)
│   ├── clustering.html                 # Hierarchical clustering & dendrogram page
│   ├── about.html                      # Project documentation & architecture page
│   └── components/                     # Reusable Jinja2 UI components
│       ├── sidebar.html                # Navigation sidebar with active route detection
│       ├── mobile_header.html          # Responsive mobile header
│       ├── city_options.html           # 42-city dropdown options
│       └── lightbox.html               # Chart zoom modal
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
| Route | Description |
| :--- | :--- |
| **`/`** | Overview & Quick Start dashboard |
| **`/regression`** | Price Valuation using Linear Regression (OLS) |
| **`/classification`** | Property Tier Classification (Decision Tree & Random Forest) |
| **`/clustering`** | Hierarchical Clustering & Dendrogram Generator |
| **`/about`** | Project Documentation, Model Mathematics & Architecture |

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

* **Classification Pipeline (Data Preparation, Decision Tree, Random Forest):**
  ```bash
  Rscript R/01_data_preparation.R
  Rscript R/02_decision_tree.R
  Rscript R/03_random_forest.R
  Rscript R/04_model_evaluation.R
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
| **Decision Tree** | Tier Classification (Multiclass) | Accuracy: **$88.44\%$**, F1: **$0.883$** |
| **Hierarchical Clustering** | Property Grouping (Unsupervised) | Ward's method on Euclidean distance |
