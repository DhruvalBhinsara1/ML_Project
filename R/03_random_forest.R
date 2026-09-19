library(randomForest)
library(caret)

dir.create("plots", showWarnings = FALSE)
dir.create("models", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

cat("Packages loaded successfully.\n")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

train_data <- read.csv("data/train_data.csv")
test_data <- read.csv("data/test_data.csv")

cat("Training rows:", nrow(train_data), "\n")
cat("Testing rows:", nrow(test_data), "\n")


# ------------------------------------------------------------
# 2. PREPARE TARGET VARIABLE
# ------------------------------------------------------------

train_data$Price_Category <- factor(
  train_data$Price_Category,
  levels = c("Budget", "Mid-Range", "Premium")
)

test_data$Price_Category <- factor(
  test_data$Price_Category,
  levels = c("Budget", "Mid-Range", "Premium")
)

cat("Target variable prepared.\n")


# ------------------------------------------------------------
# 3. SELECT FEATURES
# ------------------------------------------------------------
#
# Price_in_Lakhs and Price_per_SqFt are excluded because
# they contain price information and would cause target leakage.
#
# ID is excluded because it is only an identifier.
#
# Locality and other high-cardinality categorical variables
# are excluded to keep Random Forest training manageable.
#


rf_features <- c(
  "BHK",
  "Size_in_SqFt",
  "Year_Built",
  "Floor_No",
  "Total_Floors",
  "Age_of_Property",
  "Nearby_Schools",
  "Nearby_Hospitals"
)

rf_train <- train_data[
  ,
  c(rf_features, "Price_Category")
]

rf_test <- test_data[
  ,
  c(rf_features, "Price_Category")
]

cat(
  "Number of features used:",
  length(rf_features),
  "\n"
)


# ------------------------------------------------------------
# 4. TRAIN RANDOM FOREST
# ------------------------------------------------------------

cat("\n")
cat("========================================\n")
cat("TRAINING RANDOM FOREST\n")
cat("========================================\n")

set.seed(123)

# Subsample if dataset is very large for optimal performance
if (nrow(rf_train) > 30000) {
  sample_idx <- createDataPartition(rf_train$Price_Category, p = 30000 / nrow(rf_train), list = FALSE)
  rf_train_sample <- rf_train[sample_idx, ]
} else {
  rf_train_sample <- rf_train
}

random_forest_model <- randomForest(
  Price_Category ~ .,
  data = rf_train_sample,
  ntree = 100,
  mtry = 2,
  importance = TRUE
)

cat("\nRandom Forest trained successfully!\n")


# ------------------------------------------------------------
# 5. DISPLAY MODEL
# ------------------------------------------------------------

cat("\n")
cat("Random Forest Model:\n")

print(random_forest_model)


# ------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ------------------------------------------------------------

cat("\n")
cat("Feature Importance:\n")

print(
  importance(random_forest_model)
)


# ------------------------------------------------------------
# 7. SAVE FEATURE IMPORTANCE PLOT
# ------------------------------------------------------------

cat("\nSaving feature importance plot...\n")

png(
  "plots/rf_feature_importance.png",
  width = 1400,
  height = 1000
)

varImpPlot(
  random_forest_model,
  main = "Random Forest Feature Importance"
)

dev.off()

cat("Feature importance plot saved successfully.\n")


# ------------------------------------------------------------
# 8. MAKE PREDICTIONS
# ------------------------------------------------------------

cat("\n")
cat("Making Random Forest predictions...\n")

rf_predictions <- predict(
  random_forest_model,
  newdata = rf_test,
  type = "class"
)

cat("Predictions completed successfully.\n")


# ------------------------------------------------------------
# 9. CLASS PROBABILITIES
# ------------------------------------------------------------

cat("\nCalculating class probabilities...\n")

rf_probabilities <- predict(
  random_forest_model,
  newdata = rf_test,
  type = "prob"
)

cat("Class probabilities calculated successfully.\n")


# ------------------------------------------------------------
# 10. CONFUSION MATRIX
# ------------------------------------------------------------

rf_confusion <- confusionMatrix(
  rf_predictions,
  rf_test$Price_Category
)

cat("\n")
cat("========================================\n")
cat("RANDOM FOREST CONFUSION MATRIX\n")
cat("========================================\n")

print(rf_confusion)


# ------------------------------------------------------------
# 11. CALCULATE METRICS
# ------------------------------------------------------------

rf_accuracy <- as.numeric(
  rf_confusion$overall["Accuracy"]
)

rf_precision <- mean(
  rf_confusion$byClass[, "Pos Pred Value"],
  na.rm = TRUE
)

rf_recall <- mean(
  rf_confusion$byClass[, "Sensitivity"],
  na.rm = TRUE
)

rf_f1 <- mean(
  rf_confusion$byClass[, "F1"],
  na.rm = TRUE
)


# ------------------------------------------------------------
# 12. DISPLAY RESULTS
# ------------------------------------------------------------

cat("\n")
cat("========================================\n")
cat("RANDOM FOREST RESULTS\n")
cat("========================================\n")

cat(
  "Accuracy :",
  round(rf_accuracy, 4),
  "\n"
)

cat(
  "Precision:",
  round(rf_precision, 4),
  "\n"
)

cat(
  "Recall   :",
  round(rf_recall, 4),
  "\n"
)

cat(
  "F1 Score :",
  round(rf_f1, 4),
  "\n"
)


# ------------------------------------------------------------
# 13. SAVE MODEL
# ------------------------------------------------------------

saveRDS(
  random_forest_model,
  "models/random_forest.rds"
)

cat("\nRandom Forest model saved successfully.\n")


# ------------------------------------------------------------
# 14. SAVE PREDICTIONS
# ------------------------------------------------------------

rf_results <- data.frame(
  Actual = rf_test$Price_Category,
  Predicted = rf_predictions,
  Budget_Probability =
    rf_probabilities[, "Budget"],
  Mid_Range_Probability =
    rf_probabilities[, "Mid-Range"],
  Premium_Probability =
    rf_probabilities[, "Premium"]
)

write.csv(
  rf_results,
  "results/random_forest_predictions.csv",
  row.names = FALSE
)

cat("Random Forest predictions saved successfully.\n")


# ------------------------------------------------------------
# 15. SAVE METRICS
# ------------------------------------------------------------

rf_metrics <- data.frame(
  Model = "Random Forest",
  Accuracy = rf_accuracy,
  Precision = rf_precision,
  Recall = rf_recall,
  F1_Score = rf_f1
)

write.csv(
  rf_metrics,
  "results/random_forest_metrics.csv",
  row.names = FALSE
)

cat("Random Forest metrics saved successfully.\n")


# ------------------------------------------------------------
# 16. FINAL MESSAGE
# ------------------------------------------------------------

cat("\n")
cat("========================================\n")
cat("RANDOM FOREST COMPLETED SUCCESSFULLY\n")
cat("========================================\n")

cat("\nFiles created:\n")
cat("1. plots/rf_feature_importance.png\n")
cat("2. models/random_forest.rds\n")
cat("3. results/random_forest_predictions.csv\n")
cat("4. results/random_forest_metrics.csv\n")