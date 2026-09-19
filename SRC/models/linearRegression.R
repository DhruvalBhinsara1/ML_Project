# ============================================================
# HOUSE PRICE PREDICTION - LINEAR REGRESSION
# ============================================================

library(readr)
library(dplyr)
library(ggplot2)

# ============================================================
# 1. LOAD DATA
# ============================================================

data <- read_csv(
  "data/raw/india_housing_prices.csv",
  show_col_types = FALSE
)

cat("\n========================================\n")
cat("       HOUSE PRICE PREDICTION\n")
cat("========================================\n")

cat("Rows    :", nrow(data), "\n")
cat("Columns :", ncol(data), "\n")


# ============================================================
# 2. SELECT VALID FEATURES
# ============================================================

model_data <- data %>%
  select(
    City,
    Property_Type,
    BHK,
    Size_in_SqFt,
    Year_Built,
    Furnished_Status,
    Floor_No,
    Total_Floors,
    Age_of_Property,
    Nearby_Schools,
    Nearby_Hospitals,
    Public_Transport_Accessibility,
    Parking_Space,
    Security,
    Facing,
    Owner_Type,
    Availability_Status,
    Price_in_Lakhs
  )


# ============================================================
# 3. CLEAN DATA
# ============================================================

# Convert categorical columns to factors
categorical_cols <- c(
  "City",
  "Property_Type",
  "Furnished_Status",
  "Public_Transport_Accessibility",
  "Parking_Space",
  "Security",
  "Facing",
  "Owner_Type",
  "Availability_Status"
)

model_data[categorical_cols] <- lapply(
  model_data[categorical_cols],
  factor
)

# Remove missing values
model_data <- na.omit(model_data)

# Remove invalid target values
model_data <- model_data %>%
  filter(
    is.finite(Price_in_Lakhs),
    Price_in_Lakhs > 0,
    Size_in_SqFt > 0,
    BHK > 0
  )

cat("\nRows after cleaning:", nrow(model_data), "\n")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

set.seed(42)

train_index <- sample(
  seq_len(nrow(model_data)),
  size = floor(0.80 * nrow(model_data))
)

train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

cat("\nTraining rows:", nrow(train_data), "\n")
cat("Testing rows :", nrow(test_data), "\n")


# ============================================================
# 5. LINEAR REGRESSION
# ============================================================

cat("\nTraining model...\n")

# Log transformation of target helps reduce the effect
# of extremely expensive properties.

linear_model <- lm(
  log1p(Price_in_Lakhs) ~
    City +
    Property_Type +
    BHK +
    log1p(Size_in_SqFt) +
    Year_Built +
    Furnished_Status +
    Floor_No +
    Total_Floors +
    Age_of_Property +
    Nearby_Schools +
    Nearby_Hospitals +
    Public_Transport_Accessibility +
    Parking_Space +
    Security +
    Facing +
    Owner_Type +
    Availability_Status,
  data = train_data
)

cat("Model trained successfully!\n")


# ============================================================
# 6. PREDICTION
# ============================================================

log_prediction <- predict(
  linear_model,
  newdata = test_data
)

# Convert back from log scale to Lakhs
predicted_price <- expm1(log_prediction)

actual_price <- test_data$Price_in_Lakhs


# ============================================================
# 7. REMOVE INVALID PREDICTIONS
# ============================================================

valid <- is.finite(predicted_price)

predicted_price <- predicted_price[valid]
actual_price <- actual_price[valid]


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

MAE <- mean(
  abs(actual_price - predicted_price)
)

MSE <- mean(
  (actual_price - predicted_price)^2
)

RMSE <- sqrt(MSE)

R2 <- 1 -
  sum((actual_price - predicted_price)^2) /
  sum((actual_price - mean(actual_price))^2)

MAPE <- mean(
  abs(
    (actual_price - predicted_price) /
      actual_price
  )
) * 100


cat("\n========================================\n")
cat("       MODEL PERFORMANCE\n")
cat("========================================\n")

cat(
  "MAE  :", round(MAE, 2),
  "Lakhs\n"
)

cat(
  "MSE  :", round(MSE, 2),
  "\n"
)

cat(
  "RMSE :", round(RMSE, 2),
  "Lakhs\n"
)

cat(
  "R2   :", round(R2, 4),
  "\n"
)

cat(
  "MAPE :", round(MAPE, 2),
  "%\n"
)


# ============================================================
# 9. ACTUAL VS PREDICTED
# ============================================================

set.seed(42)

plot_size <- min(10000, length(actual_price))

plot_index <- sample(
  seq_along(actual_price),
  plot_size
)

plot_data <- data.frame(
  Actual = actual_price[plot_index],
  Predicted = predicted_price[plot_index]
)

graph <- ggplot(
  plot_data,
  aes(
    x = Actual,
    y = Predicted
  )
) +
  geom_point(alpha = 0.35) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  labs(
    title = "Actual vs Predicted House Prices",
    x = "Actual Price (Lakhs)",
    y = "Predicted Price (Lakhs)"
  ) +
  theme_minimal()


# ============================================================
# 10. SAVE GRAPH
# ============================================================

dir.create(
  "outputs",
  showWarnings = FALSE
)

ggsave(
  "outputs/linearRegression_actual_vs_predicted.png",
  graph,
  width = 8,
  height = 6,
  dpi = 300
)

cat("\nGraph saved successfully!\n")


# ============================================================
# 11. SAVE MODEL
# ============================================================

dir.create(
  "models",
  showWarnings = FALSE
)

saveRDS(
  linear_model,
  "models/linearRegression.rds"
)

cat("Model saved successfully!\n")


# ============================================================
# 12. HOUSE PRICE PREDICTION DEMO
# ============================================================

cat("\n========================================\n")
cat("       NEW HOUSE PRICE PREDICTION\n")
cat("========================================\n")

if (interactive()) {
  # ---------- Numeric input ----------
  BHK_input <- as.numeric(readline("Enter BHK: "))
  Size_input <- as.numeric(readline("Enter Size in SqFt: "))
  Year_input <- as.numeric(readline("Enter Year Built: "))
  Floor_input <- as.numeric(readline("Enter Floor Number: "))
  TotalFloors_input <- as.numeric(readline("Enter Total Floors: "))
  Age_input <- as.numeric(readline("Enter Age of Property: "))
  Schools_input <- as.numeric(readline("Enter Nearby Schools: "))
  Hospitals_input <- as.numeric(readline("Enter Nearby Hospitals: "))

  # ---------- Categorical input ----------
  get_category <- function(name, levels_available) {
    cat("\nAvailable", name, "values:\n")
    print(levels_available)
    value <- trimws(readline(paste0("Enter ", name, ": ")))
    while (!(value %in% levels_available)) {
      cat("\nInvalid value. Choose one of:\n")
      print(levels_available)
      value <- trimws(readline(paste0("Enter ", name, ": ")))
    }
    factor(value, levels = levels_available)
  }

  City_input <- get_category("City", levels(model_data$City))
  Property_input <- get_category("Property Type", levels(model_data$Property_Type))
  Furnished_input <- get_category("Furnished Status", levels(model_data$Furnished_Status))
  Transport_input <- get_category("Public Transport Accessibility", levels(model_data$Public_Transport_Accessibility))
  Parking_input <- get_category("Parking Space", levels(model_data$Parking_Space))
  Security_input <- get_category("Security", levels(model_data$Security))
  Facing_input <- get_category("Facing", levels(model_data$Facing))
  Owner_input <- get_category("Owner Type", levels(model_data$Owner_Type))
  Availability_input <- get_category("Availability Status", levels(model_data$Availability_Status))

  new_house <- data.frame(
    City = City_input,
    Property_Type = Property_input,
    BHK = BHK_input,
    Size_in_SqFt = Size_input,
    Year_Built = Year_input,
    Furnished_Status = Furnished_input,
    Floor_No = Floor_input,
    Total_Floors = TotalFloors_input,
    Age_of_Property = Age_input,
    Nearby_Schools = Schools_input,
    Nearby_Hospitals = Hospitals_input,
    Public_Transport_Accessibility = Transport_input,
    Parking_Space = Parking_input,
    Security = Security_input,
    Facing = Facing_input,
    Owner_Type = Owner_input,
    Availability_Status = Availability_input
  )
} else {
  # Default sample for non-interactive execution
  cat("Running sample prediction (3 BHK, 1800 SqFt, Pune)...\n")
  new_house <- test_data[1, ]
  new_house$BHK <- 3
  new_house$Size_in_SqFt <- 1800
  new_house$Age_of_Property <- 5
  new_house$City <- factor("Pune", levels = levels(model_data$City))
}

# ============================================================
# 13. PREDICT & DISPLAY FINAL PRICE
# ============================================================

new_log_prediction <- predict(
  linear_model,
  newdata = new_house
)

final_price <- expm1(
  new_log_prediction
)

cat("\n========================================\n")
cat("       PREDICTION RESULT\n")
cat("========================================\n")

cat(
  "\nPredicted House Price:",
  round(final_price[1], 2),
  "Lakhs\n"
)

cat(
  "\nApproximate price: ₹",
  format(round(final_price[1] * 100000, 0), big.mark = ","),
  "\n"
)

cat("\n========================================\n")