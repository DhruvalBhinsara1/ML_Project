# src/predict_classification.R
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript src/predict_classification.R <size> <bhk> <age> <city>")
}

size <- as.numeric(args[1])
bhk <- as.numeric(args[2])
age <- as.numeric(args[3])
city <- args[4]

models_dir <- "models"
dir.create(models_dir, showWarnings = FALSE)
model_path_dt <- file.path(models_dir, "decision_tree.rds")
model_path_rf <- file.path(models_dir, "random_forest.rds")

if (!file.exists(model_path_dt) || !file.exists(model_path_rf)) {
  source("src/classification.R")
}

dt_model <- readRDS(model_path_dt)
rf_model <- readRDS(model_path_rf)

# Extract levels from model
city_levels <- rf_model$forest$xlevels$City

# Fallback for trees if it isn't set
if (is.null(city_levels)) {
  city_levels <- c('Chennai', 'Pune', 'Ludhiana', 'Jodhpur', 'Jaipur', 'Durgapur', 'Coimbatore', 'Bilaspur', 'New Delhi', 'Ranchi', 'Warangal', 'Bangalore', 'Nagpur', 'Lucknow', 'Silchar', 'Dehradun', 'Noida', 'Gaya', 'Jamshedpur', 'Ahmedabad', 'Hyderabad', 'Faridabad', 'Amritsar', 'Kolkata', 'Dwarka', 'Vishakhapatnam', 'Bhopal', 'Indore', 'Haridwar', 'Mysore', 'Patna', 'Raipur', 'Vijayawada', 'Trivandrum', 'Kochi', 'Surat', 'Gurgaon', 'Mangalore', 'Cuttack', 'Bhubaneswar', 'Guwahati', 'Mumbai')
}

# Handle unseen categories gracefully
if (!city %in% city_levels) city <- city_levels[1]

new_data <- data.frame(
  Size_in_SqFt = size, 
  BHK = bhk, 
  Age_of_Property = age, 
  City = factor(city, levels = city_levels)
)

dt_pred <- predict(dt_model, new_data, type = "class")
rf_pred <- predict(rf_model, new_data)

cat("\nRESULT:", as.character(dt_pred[1]), "|", as.character(rf_pred[1]), "\n")
