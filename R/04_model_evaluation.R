library(randomForest)
library(caret)

cat("Packages loaded successfully.\n")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

train_data <- read.csv("data/train_data.csv")
test_data <- read.csv("data/test_data.csv")

cat("Full training rows:", nrow(train_data), "\n")
cat("Testing rows:", nrow(test_data), "\n")


# ------------------------------------------------------------
# 2. PREPARE TARGET
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

rf_train_full <- train_data[
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
# 4. CREATE STRATIFIED SAMPLE
# ------------------------------------------------------------

set.seed(123)

sample_size <- 30000

sample_index <- createDataPartition(
  rf_train_full$Price_Category,
  p = sample_size / nrow(rf_train_full),
  list = FALSE
)

rf_train <- rf_train_full[sample_index, ]

cat(
  "Random Forest training sample:",
  nrow(rf_train),
  "rows\n"
)


# ------------------------------------------------------------
# 5. TRAIN RANDOM FOREST
# ------------------------------------------------------------

cat("\n")
cat("========================================\n")
cat("TRAINING RANDOM FOREST\n")
cat("========================================\n")

set.seed(123)

random_forest_model <- randomForest(
  Price_Category ~ .,
  data = rf_train,
  ntree = 50,
  mtry = 2,
  importance = TRUE
)

cat("\nRandom Forest trained successfully!\n")


# ------------------------------------------------------------
# 6. DISPLAY MODEL
# ------------------------------------------------------------

cat("\n")
cat("Random Forest Model:\n")

print(random_forest_model)


# ------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ------------------------------------------------------------

cat("\n")
cat("Feature Importance:\n")

print(
  importance(random_forest_model)
)


# ------------------------------------------------------------
# 8. SAVE FEATURE IMPORTANCE PLOT
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
# 9. MAKE PREDICTIONS
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
# 10. CLASS PROBABILITIES
# ------------------------------------------------------------

cat("\nCalculating class probabilities...\n")

rf_probabilities <- predict(
  random_forest_model,
  newdata = rf_test,
  type = "prob"
)

cat("Class probabilities calculated successfully.\n")


# ------------------------------------------------------------
# 11. CONFUSION MATRIX
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
# 12. CALCULATE METRICS
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
# 13. DISPLAY RESULTS
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
# 14. SAVE MODEL
# ------------------------------------------------------------

saveRDS(
  random_forest_model,
  "models/random_forest.rds"
)

cat("\nRandom Forest model saved successfully.\n")


# ------------------------------------------------------------
# 15. SAVE PREDICTIONS
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
# 16. SAVE METRICS
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
# 17. FINAL MESSAGE
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