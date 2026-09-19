# src/classification.R

# Install required packages if not present
if (!require(rpart)) install.packages('rpart', repos='http://cran.us.r-project.org')
if (!require(randomForest)) install.packages('randomForest', repos='http://cran.us.r-project.org')
if (!require(caret)) install.packages('caret', repos='http://cran.us.r-project.org')

library(rpart)
library(randomForest)
library(caret)

# Source the preprocessing script
source("src/preprocessing.R")

# Load and preprocess data
cat("Loading and preprocessing data...\n")
df <- preprocess_data()

# We only need the features and Category (target)
features <- c("Size_in_SqFt", "BHK", "Age_of_Property", "City")
target <- "Category"

# Create formula
formula <- as.formula(paste(target, "~", paste(features, collapse = " + ")))

# Train/Test Split
set.seed(42)
# Sample down to 5000 rows max to make training fast for the demo
if (nrow(df) > 5000) {
    df <- df[sample(nrow(df), 5000), ]
}

trainIndex <- createDataPartition(df$Category, p = 0.8, list = FALSE)
train_data <- df[trainIndex, ]
test_data  <- df[-trainIndex, ]

cat("\n=========================================\n")
cat("      DECISION TREE CLASSIFICATION       \n")
cat("=========================================\n")

# Train Decision Tree
dt_model <- rpart(formula, data = train_data, method = "class")

# Predict
dt_preds <- predict(dt_model, test_data, type = "class")

# Metrics
dt_cm <- confusionMatrix(dt_preds, test_data$Category)
cat("Decision Tree Accuracy: ", round(dt_cm$overall['Accuracy'] * 100, 2), "%\n")
print(dt_cm$table)

# For precision, recall, f1, caret provides them in `byClass`
print(dt_cm$byClass[, c("Precision", "Recall", "F1")])

# Save Decision Tree Plot
if (!require(rpart.plot)) install.packages('rpart.plot', repos='http://cran.us.r-project.org')
library(rpart.plot)
dir.create("static", showWarnings = FALSE)
png("static/decision_tree.png", width = 800, height = 600)
rpart.plot(dt_model, main = "Real Estate Classification Tree", type = 4, extra = 104)
dev.off()



cat("\n=========================================\n")
cat("       RANDOM FOREST CLASSIFICATION      \n")
cat("=========================================\n")

# Train Random Forest
# Using a smaller ntree for speed during testing
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

# Save models for the web app (if needed later)
dir.create("models", showWarnings = FALSE)
saveRDS(dt_model, "models/decision_tree.rds")
saveRDS(rf_model, "models/random_forest.rds")
cat("\nModels saved to 'models/' directory.\n")
