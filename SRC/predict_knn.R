# SRC/predict_knn.R
suppressPackageStartupMessages({
  library(caret)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 8) {
  stop("Usage: Rscript SRC/predict_knn.R <bhk> <size> <year_built> <floor_no> <total_floors> <age> <nearby_schools> <nearby_hospitals>")
}

bhk <- as.numeric(args[1])
size <- as.numeric(args[2])
year_built <- as.numeric(args[3])
floor_no <- as.numeric(args[4])
total_floors <- as.numeric(args[5])
age <- as.numeric(args[6])
nearby_schools <- as.numeric(args[7])
nearby_hospitals <- as.numeric(args[8])

model_path <- "models/knn_model.rds"
if (!file.exists(model_path)) {
  if (file.exists("../models/knn_model.rds")) {
    model_path <- "../models/knn_model.rds"
  } else {
    stop("KNN model file not found at models/knn_model.rds")
  }
}

knn_bundle <- readRDS(model_path)

input_df <- data.frame(
  BHK = bhk,
  Size_in_SqFt = size,
  Year_Built = year_built,
  Floor_No = floor_no,
  Total_Floors = total_floors,
  Age_of_Property = age,
  Nearby_Schools = nearby_schools,
  Nearby_Hospitals = nearby_hospitals
)

scaled_input <- predict(knn_bundle$preprocessing, input_df)
pred_class <- predict(knn_bundle$model, scaled_input, type = "class")
pred_probs <- predict(knn_bundle$model, scaled_input, type = "prob")

budget_prob <- if ("Budget" %in% colnames(pred_probs)) round(pred_probs[1, "Budget"] * 100, 1) else 0.0
mid_prob <- if ("Mid-Range" %in% colnames(pred_probs)) round(pred_probs[1, "Mid-Range"] * 100, 1) else 0.0
premium_prob <- if ("Premium" %in% colnames(pred_probs)) round(pred_probs[1, "Premium"] * 100, 1) else 0.0
best_k <- if (!is.null(knn_bundle$best_k)) knn_bundle$best_k else 11

cat("\nRESULT:", as.character(pred_class[1]), "|", budget_prob, "|", mid_prob, "|", premium_prob, "|", best_k, "\n")
