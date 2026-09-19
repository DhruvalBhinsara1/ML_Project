library(rpart)
library(rpart.plot)
library(caret)

cat("Packages loaded successfully.\n")

train_data <- read.csv("data/train_data.csv")
test_data <- read.csv("data/test_data.csv")

cat("Training rows:", nrow(train_data), "\n")
cat("Testing rows:", nrow(test_data), "\n")

train_data$Price_Category <- factor(
  train_data$Price_Category,
  levels = c("Budget", "Mid-Range", "Premium")
)

test_data$Price_Category <- factor(
  test_data$Price_Category,
  levels = c("Budget", "Mid-Range", "Premium")
)

cat("Target variable prepared.\n")

tree_features <- c(
  "BHK",
  "Size_in_SqFt",
  "Year_Built",
  "Floor_No",
  "Total_Floors",
  "Age_of_Property",
  "Nearby_Schools",
  "Nearby_Hospitals"
)

tree_train <- train_data[, c(tree_features, "Price_Category")]
tree_test <- test_data[, c(tree_features, "Price_Category")]

cat(
  "Number of features used:",
  length(tree_features),
  "\n"
)

cat("\nTraining Decision Tree...\n")

set.seed(123)

decision_tree_model <- rpart(
  Price_Category ~ .,
  data = tree_train,
  method = "class",
  control = rpart.control(
  cp = 0.0001,
  minsplit = 100,
  maxdepth = 8,
  xval = 0
)
)

cat("\nDecision Tree trained successfully!\n")

cat("\nDecision Tree Model:\n")
print(decision_tree_model)

cat("\nVariable Importance:\n")

if (!is.null(decision_tree_model$variable.importance)) {
  print(decision_tree_model$variable.importance)
} else {
  cat("No variable importance available.\n")
}

cat("\nSaving Decision Tree plot...\n")

png(
  "plots/decision_tree.png",
  width = 1400,
  height = 1000
)

rpart.plot(
  decision_tree_model,
  main = "Decision Tree - House Price Classification",
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  box.palette = "Blues"
)

dev.off()

cat("Decision Tree plot saved successfully.\n")

cat("\nMaking test predictions...\n")

dt_predictions <- predict(
  decision_tree_model,
  newdata = tree_test,
  type = "class"
)

cat("Predictions completed successfully.\n")

cat("\nCalculating class probabilities...\n")

dt_probabilities <- predict(
  decision_tree_model,
  newdata = tree_test,
  type = "prob"
)

cat("Class probabilities calculated successfully.\n")

dt_confusion <- confusionMatrix(
  dt_predictions,
  tree_test$Price_Category
)

cat("\nDecision Tree Confusion Matrix:\n")
print(dt_confusion)

dt_accuracy <- as.numeric(
  dt_confusion$overall["Accuracy"]
)

dt_precision <- mean(
  dt_confusion$byClass[, "Pos Pred Value"],
  na.rm = TRUE
)

dt_recall <- mean(
  dt_confusion$byClass[, "Sensitivity"],
  na.rm = TRUE
)

dt_f1 <- mean(
  dt_confusion$byClass[, "F1"],
  na.rm = TRUE
)

cat("\nDecision Tree Results:\n")

cat(
  "Accuracy :",
  round(dt_accuracy, 4),
  "\n"
)

cat(
  "Precision:",
  round(dt_precision, 4),
  "\n"
)

cat(
  "Recall   :",
  round(dt_recall, 4),
  "\n"
)

cat(
  "F1 Score :",
  round(dt_f1, 4),
  "\n"
)

saveRDS(
  decision_tree_model,
  "models/decision_tree.rds"
)

cat("\nDecision Tree model saved successfully.\n")

dt_results <- data.frame(
  Actual = tree_test$Price_Category,
  Predicted = dt_predictions,
  Budget_Probability = dt_probabilities[, "Budget"],
  Mid_Range_Probability = dt_probabilities[, "Mid-Range"],
  Premium_Probability = dt_probabilities[, "Premium"]
)

write.csv(
  dt_results,
  "results/decision_tree_predictions.csv",
  row.names = FALSE
)

cat("Decision Tree predictions saved successfully.\n")

dt_metrics <- data.frame(
  Model = "Decision Tree",
  Accuracy = dt_accuracy,
  Precision = dt_precision,
  Recall = dt_recall,
  F1_Score = dt_f1
)

write.csv(
  dt_metrics,
  "results/decision_tree_metrics.csv",
  row.names = FALSE
)

cat("Decision Tree metrics saved successfully.\n")

cat("\n========================================\n")
cat("DECISION TREE COMPLETED SUCCESSFULLY\n")
cat("========================================\n")