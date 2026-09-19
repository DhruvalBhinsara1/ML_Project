# SRC/classification.R

suppressPackageStartupMessages({
  library(rpart)
  library(randomForest)
  library(caret)
  library(rpart.plot)
})

# Source the preprocessing script
if (file.exists("SRC/preprocessing.R")) {
  source("SRC/preprocessing.R")
} else if (file.exists("src/preprocessing.R")) {
  source("src/preprocessing.R")
} else {
  source("preprocessing.R")
}

# Load and preprocess data
cat("Loading and preprocessing data...\n")
df <- preprocess_data()

# Calculate city price scores to avoid 2^41 factor explosion in tree splits
city_scores <- tapply(df$Price_in_Lakhs, df$City, mean, na.rm = TRUE)
default_city_score <- mean(df$Price_in_Lakhs, na.rm = TRUE)
df$City_Score <- as.numeric(city_scores[as.character(df$City)])
df$City_Score[is.na(df$City_Score)] <- default_city_score

# Features and target for Classification
features <- c("Size_in_SqFt", "BHK", "Age_of_Property", "City_Score")
target <- "Category"
formula <- as.formula(paste(target, "~", paste(features, collapse = " + ")))

# Train/Test Split
set.seed(42)
sample_size <- min(10000, nrow(df))
df_sample <- df[sample(nrow(df), sample_size), ]

trainIndex <- createDataPartition(df_sample$Category, p = 0.8, list = FALSE)
train_data <- df_sample[trainIndex, ]
test_data  <- df_sample[-trainIndex, ]

cat("\n=========================================\n")
cat("      DECISION TREE CLASSIFICATION       \n")
cat("=========================================\n")

# Train Decision Tree
dt_model <- rpart(formula, data = train_data, method = "class", control = rpart.control(cp = 0.01))

# Predict
dt_preds <- predict(dt_model, test_data, type = "class")

# Metrics
dt_cm <- confusionMatrix(dt_preds, test_data$Category)
cat("Decision Tree Accuracy: ", round(dt_cm$overall['Accuracy'] * 100, 2), "%\n")
print(dt_cm$table)
print(dt_cm$byClass[, c("Precision", "Recall", "F1")])

# Save Decision Tree Plots
dir.create("static", showWarnings = FALSE)
dir.create("plots", showWarnings = FALSE)

png("static/decision_tree.png", width = 800, height = 600)
rpart.plot(dt_model, main = "Real Estate Classification Tree", type = 4, extra = 104, box.palette = "Blues")
dev.off()

png("plots/decision_tree.png", width = 800, height = 600)
rpart.plot(dt_model, main = "Real Estate Classification Tree", type = 4, extra = 104, box.palette = "Blues")
dev.off()

cat("\n=========================================\n")
cat("       RANDOM FOREST CLASSIFICATION      \n")
cat("=========================================\n")

# Train Random Forest
rf_model <- randomForest(formula, data = train_data, ntree = 100, importance = TRUE)

# Predict
rf_preds <- predict(rf_model, test_data)

# Metrics
rf_cm <- confusionMatrix(rf_preds, test_data$Category)
cat("Random Forest Accuracy: ", round(rf_cm$overall['Accuracy'] * 100, 2), "%\n")
print(rf_cm$table)
print(rf_cm$byClass[, c("Precision", "Recall", "F1")])

cat("\n--- Feature Importance ---\n")
print(importance(rf_model))

# Save models and city mapping for the web app and inference
dir.create("models", showWarnings = FALSE)
saveRDS(dt_model, "models/dt_classifier.rds")
saveRDS(rf_model, "models/rf_classifier.rds")
saveRDS(dt_model, "models/decision_tree.rds")
saveRDS(rf_model, "models/random_forest.rds")
saveRDS(list(city_scores = city_scores, default_score = default_city_score), "models/city_mapping.rds")
cat("\nModels and city mapping saved to 'models/' directory.\n")
