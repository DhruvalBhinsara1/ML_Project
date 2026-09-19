# ML_Project

House price analysis in R: linear regression, buyer classification, and
hierarchical clustering, plus a Flask web UI over the trained models.

## Structure

```
SRC/models/linearRegression.R   # standalone linear regression script
ml/clustering.R                 # standalone hierarchical clustering script
R/                               # data prep, decision tree, random forest, evaluation
src/                             # Flask app's R scripts (train/predict per model)
app/app.py                       # Flask web app
templates/, static/              # Flask app UI + assets
models/                          # saved .rds models
outputs/, plots/                 # generated plots
data/raw/                        # dataset
```

## Requirements

- R (4.x) with packages: `readr`, `dplyr`, `ggplot2`

```r
install.packages(c("readr", "dplyr", "ggplot2"))
```

- Python 3 with Flask, for the web app

```bash
pip install -r requirements.txt
```

## Usage

Standalone R scripts:

```bash
Rscript SRC/models/linearRegression.R
Rscript ml/clustering.R
```

Flask web app:

```bash
python app/app.py
```

Then open `http://127.0.0.1:5000`.
