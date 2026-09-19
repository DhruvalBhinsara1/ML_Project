# ============================================================
# 05_knn.R
# K-Nearest Neighbors Classification
# India House Price Prediction Project
# ============================================================

library(caret)

# ============================================================
# 1. LOAD TRAINING AND TEST DATA
# ============================================================

train_data <- read.csv("data/train_data.csv")
test_data <- read.csv("data/test_data.csv")

cat("Training rows:", nrow(train_data), "\n")
cat("Testing rows:", nrow(test_data), "\n")

# Make target a factor
train_data$Price_Category <- factor(
  train_data$Price_Category,
  levels = c("Budget", "Mid-Range", "Premium")
)

test_data$Price_Category <- factor(
  test_data$Price_Category,
  levels = c("Budget", "Mid-Range", "Premium")
)

# ============================================================
# 2. SELECT FEATURES
# ============================================================

features <- c(
  "BHK",
  "Size_in_SqFt",
  "Year_Built",
  "Floor_No",
  "Total_Floors",
  "Age_of_Property",
  "Nearby_Schools",
  "Nearby_Hospitals"
)

train_x <- train_data[, features]
test_x <- test_data[, features]

train_y <- train_data$Price_Category
test_y <- test_data$Price_Category

# ============================================================
# 3. SCALE FEATURES
# ============================================================
# KNN uses distance, so all variables should be standardized.

preprocess <- preProcess(
  train_x,
  method = c("center", "scale")
)

train_x_scaled <- predict(preprocess, train_x)
test_x_scaled <- predict(preprocess, test_x)

cat("Feature scaling completed.\n")

# ============================================================
# 4. TEST DIFFERENT K VALUES
# ============================================================

k_values <- c(3, 5, 7, 9, 11)

k_results <- data.frame(
  K = integer(),
  Accuracy = numeric()
)

for (k in k_values) {

  cat("Testing K =", k, "\n")

  model_temp <- knn3(
    x = train_x_scaled,
    y = train_y,
    k = k
  )

  predictions_temp <- predict(
    model_temp,
    test_x_scaled,
    type = "class"
  )

  cm_temp <- confusionMatrix(
    predictions_temp,
    test_y
  )

  k_results <- rbind(
    k_results,
    data.frame(
      K = k,
      Accuracy = as.numeric(
        cm_temp$overall["Accuracy"]
      )
    )
  )
}

cat("\nK vs Accuracy:\n")
print(k_results)

# ============================================================
# 5. SELECT BEST K
# ============================================================

best_k <- k_results$K[
  which.max(k_results$Accuracy)
]

cat("\nBest K =", best_k, "\n")

# ============================================================
# 6. TRAIN FINAL KNN MODEL
# ============================================================

knn_model <- knn3(
  x = train_x_scaled,
  y = train_y,
  k = best_k
)

cat("Final KNN model trained.\n")

# ============================================================
# 7. CLASS PREDICTIONS
# ============================================================

knn_predictions <- predict(
  knn_model,
  test_x_scaled,
  type = "class"
)

# ============================================================
# 8. CLASS PROBABILITIES
# ============================================================

knn_probabilities <- predict(
  knn_model,
  test_x_scaled,
  type = "prob"
)

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm <- confusionMatrix(
  knn_predictions,
  test_y
)

cat("\n====================================\n")
cat("KNN CONFUSION MATRIX\n")
cat("====================================\n")

print(cm$table)

# ============================================================
# 10. CALCULATE METRICS
# ============================================================

accuracy <- as.numeric(
  cm$overall["Accuracy"]
)

precision <- mean(
  cm$byClass[, "Pos Pred Value"],
  na.rm = TRUE
)

recall <- mean(
  cm$byClass[, "Sensitivity"],
  na.rm = TRUE
)

f1_values <- 2 *
  cm$byClass[, "Pos Pred Value"] *
  cm$byClass[, "Sensitivity"] /
  (
    cm$byClass[, "Pos Pred Value"] +
    cm$byClass[, "Sensitivity"]
  )

f1 <- mean(
  f1_values,
  na.rm = TRUE
)

knn_metrics <- data.frame(
  Model = "KNN",
  Best_K = best_k,
  Accuracy = accuracy,
  Precision = precision,
  Recall = recall,
  F1_Score = f1
)

cat("\n====================================\n")
cat("KNN METRICS\n")
cat("====================================\n")

print(knn_metrics)

# ============================================================
# 11. SAVE PREDICTIONS + PROBABILITIES
# ============================================================

knn_predictions_output <- data.frame(
  Actual = test_y,
  Predicted = knn_predictions,
  Budget_Probability = knn_probabilities[, "Budget"],
  Mid_Range_Probability = knn_probabilities[, "Mid-Range"],
  Premium_Probability = knn_probabilities[, "Premium"]
)

write.csv(
  knn_predictions_output,
  "results/knn_predictions.csv",
  row.names = FALSE
)

cat("\nKNN predictions saved.\n")

# ============================================================
# 12. SAVE METRICS
# ============================================================

write.csv(
  knn_metrics,
  "results/knn_metrics.csv",
  row.names = FALSE
)

cat("KNN metrics saved.\n")

# ============================================================
# 13. SAVE MODEL
# ============================================================

saveRDS(
  list(
    model = knn_model,
    preprocessing = preprocess,
    features = features,
    best_k = best_k
  ),
  "models/knn_model.rds"
)

cat("KNN model saved.\n")

# ============================================================
# 14. SAVE K VS ACCURACY PLOT
# ============================================================

png(
  "plots/knn_accuracy_vs_k.png",
  width = 900,
  height = 600
)

plot(
  k_results$K,
  k_results$Accuracy,
  type = "b",
  pch = 19,
  xlab = "K (Number of Neighbors)",
  ylab = "Accuracy",
  main = "KNN Accuracy for Different K Values"
)

grid()

dev.off()

cat("KNN accuracy plot saved.\n")

# ============================================================
# 15. FINAL MESSAGE
# ============================================================

cat("\n====================================\n")
cat("KNN CLASSIFICATION COMPLETED\n")
cat("====================================\n")

cat("Best K:", best_k, "\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1 Score:", round(f1, 4), "\n")

cat("\nFiles created:\n")
cat("models/knn_model.rds\n")
cat("results/knn_metrics.csv\n")
cat("results/knn_predictions.csv\n")
cat("plots/knn_accuracy_vs_k.png\n")