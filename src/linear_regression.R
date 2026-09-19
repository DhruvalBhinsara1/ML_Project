# SRC/linear_regression.R

if (file.exists("SRC/preprocessing.R")) {
  source("SRC/preprocessing.R")
} else if (file.exists("src/preprocessing.R")) {
  source("src/preprocessing.R")
} else {
  source("preprocessing.R")
}

cat("Loading and preprocessing data for Linear Regression...\n")
df <- preprocess_data()

# We only need the features and target for Regression
features <- c("Size_in_SqFt", "BHK", "Age_of_Property", "City")
target <- "Price_in_Lakhs"

# Train/Test Split
set.seed(42)
sample_index <- sample(seq_len(nrow(df)), size = 0.8 * nrow(df))
train_data <- df[sample_index, ]
test_data  <- df[-sample_index, ]

formula <- as.formula(paste(target, "~", paste(features, collapse = " + ")))

cat("\n=========================================\n")
cat("            LINEAR REGRESSION            \n")
cat("=========================================\n")

# Train Model
lr_model <- lm(formula, data = train_data)

# Predict
preds <- predict(lr_model, test_data)

# Calculate Metrics (MAE, MSE, RMSE, R2)
actual <- test_data$Price_in_Lakhs

mae <- mean(abs(actual - preds))
mse <- mean((actual - preds)^2)
rmse <- sqrt(mse)

# R-squared manually calculated
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss

cat("Mean Absolute Error (MAE):", round(mae, 2), "\n")
cat("Mean Squared Error (MSE):", round(mse, 2), "\n")
cat("Root Mean Squared Error (RMSE):", round(rmse, 2), "\n")
cat("R-squared (R2):", round(rsq, 4), "\n")

# Save model
dir.create("models", showWarnings = FALSE)
saveRDS(lr_model, "models/linear_regression.rds")
cat("\nLinear Regression model saved to 'models/linear_regression.rds'.\n")
