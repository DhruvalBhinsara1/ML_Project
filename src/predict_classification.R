# SRC/predict_classification.R
suppressPackageStartupMessages({
  library(rpart)
  library(randomForest)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript SRC/predict_classification.R <size> <bhk> <age> <city>")
}

size <- as.numeric(args[1])
bhk <- as.numeric(args[2])
age <- as.numeric(args[3])
city <- trimws(args[4])

models_dir <- "models"
dir.create(models_dir, showWarnings = FALSE)
model_path_dt <- file.path(models_dir, "dt_classifier.rds")
model_path_rf <- file.path(models_dir, "rf_classifier.rds")
mapping_path  <- file.path(models_dir, "city_mapping.rds")

if (!file.exists(model_path_dt) || !file.exists(model_path_rf) || !file.exists(mapping_path)) {
  if (file.exists("SRC/classification.R")) {
    source("SRC/classification.R")
  } else if (file.exists("src/classification.R")) {
    source("src/classification.R")
  }
}

dt_model <- readRDS(model_path_dt)
rf_model <- readRDS(model_path_rf)
city_map <- readRDS(mapping_path)

city_score <- city_map$city_scores[[city]]
if (is.null(city_score) || is.na(city_score)) {
  city_score <- city_map$default_score
}

new_data <- data.frame(
  Size_in_SqFt = size, 
  BHK = bhk, 
  Age_of_Property = age, 
  City_Score = city_score
)

dt_pred <- predict(dt_model, new_data, type = "class")
rf_pred <- predict(rf_model, new_data)

cat("\nRESULT:", as.character(dt_pred[1]), "|", as.character(rf_pred[1]), "\n")
