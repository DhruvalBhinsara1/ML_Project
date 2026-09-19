# SRC/predict_regression.R
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript SRC/predict_regression.R <size> <bhk> <age> <city>")
}

size <- as.numeric(args[1])
bhk <- as.numeric(args[2])
age <- as.numeric(args[3])
city <- trimws(args[4])

models_dir <- "models"
dir.create(models_dir, showWarnings = FALSE)
model_path <- file.path(models_dir, "linear_regression.rds")

if (!file.exists(model_path)) {
  if (file.exists("SRC/linear_regression.R")) {
    source("SRC/linear_regression.R")
  } else if (file.exists("src/linear_regression.R")) {
    source("src/linear_regression.R")
  }
}

lr_model <- readRDS(model_path)

# Extract levels from model
city_levels <- lr_model$xlevels$City

# Handle unseen categories gracefully
if (!city %in% city_levels) city <- city_levels[1]

new_data <- data.frame(
  Size_in_SqFt = size, 
  BHK = bhk, 
  Age_of_Property = age, 
  City = factor(city, levels = city_levels)
)
pred <- predict(lr_model, new_data)

if (pred < 5) {
  pred <- 5.00
}

cat("\nRESULT:", round(pred, 2), "\n")
