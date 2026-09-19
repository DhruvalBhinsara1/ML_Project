# ML_Project

R scripts for house price analysis: linear regression + hierarchical clustering.

## Structure

```
SRC/models/linearRegression.R   # linear regression on house prices
ml/clustering.R                 # hierarchical clustering + dendrogram
models/linearRegression.rds     # saved regression model
outputs/                        # generated plots (dendrogram, actual vs predicted)
```

## Requirements

- R (4.x)
- Packages: `readr`, `dplyr`, `ggplot2`

```r
install.packages(c("readr", "dplyr", "ggplot2"))
```

## Data

Scripts expect an India housing prices CSV, not included in this repo:

- `SRC/models/linearRegression.R` reads `DATA/RAW/india_housing_prices.csv`
- `ml/clustering.R` reads `data/india_housing_prices.csv`

Place the dataset at the matching path before running.

## Usage

```bash
Rscript SRC/models/linearRegression.R
Rscript ml/clustering.R
```
