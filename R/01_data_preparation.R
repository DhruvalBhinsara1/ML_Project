# ============================================================
# INDIA HOUSE PRICE PREDICTION
# MEMBER 3 - CLASSIFICATION
# 01 - DATA PREPARATION
# ============================================================


# ------------------------------------------------------------
# 1. LOAD LIBRARIES
# ------------------------------------------------------------

library(tidyverse)
library(caret)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

data <- read.csv("data/india_housing_prices.csv")


# ------------------------------------------------------------
# 3. BASIC DATA INFORMATION
# ------------------------------------------------------------

cat("Number of rows:", nrow(data), "\n")
cat("Number of columns:", ncol(data), "\n")

cat("\nColumn names:\n")
print(names(data))


# ------------------------------------------------------------
# 4. CHECK DATA STRUCTURE
# ------------------------------------------------------------

cat("\nData structure:\n")
str(data)


# ------------------------------------------------------------
# 5. CHECK MISSING VALUES
# ------------------------------------------------------------

cat("\nMissing values:\n")
print(colSums(is.na(data)))


# ------------------------------------------------------------
# 6. PRICE SUMMARY
# ------------------------------------------------------------

cat("\nPrice summary:\n")
print(summary(data$Price_in_Lakhs))


# ------------------------------------------------------------
# 7. CREATE PRICE CATEGORY BOUNDARIES
# ------------------------------------------------------------

lower_boundary <- quantile(
  data$Price_in_Lakhs,
  probs = 1 / 3,
  na.rm = TRUE
)

upper_boundary <- quantile(
  data$Price_in_Lakhs,
  probs = 2 / 3,
  na.rm = TRUE
)

cat("\nBudget / Mid-Range boundary:",
    lower_boundary, "\n")

cat("Mid-Range / Premium boundary:",
    upper_boundary, "\n")


# ------------------------------------------------------------
# 8. CREATE PRICE CATEGORY
# ------------------------------------------------------------

data <- data %>%
  mutate(
    Price_Category = case_when(
      Price_in_Lakhs <= lower_boundary ~ "Budget",
      Price_in_Lakhs <= upper_boundary ~ "Mid-Range",
      TRUE ~ "Premium"
    )
  )


# Convert target to factor

data$Price_Category <- factor(
  data$Price_Category,
  levels = c(
    "Budget",
    "Mid-Range",
    "Premium"
  )
)


# ------------------------------------------------------------
# 9. CHECK CLASS DISTRIBUTION
# ------------------------------------------------------------

cat("\nPrice category distribution:\n")

print(
  table(data$Price_Category)
)

cat("\nPrice category percentages:\n")

print(
  round(
    prop.table(table(data$Price_Category)) * 100,
    2
  )
)


# ------------------------------------------------------------
# 10. REMOVE VARIABLES NOT USED FOR CLASSIFICATION
# ------------------------------------------------------------

classification_data <- data %>%
  select(
    -ID,
    -Price_in_Lakhs,
    -Price_per_SqFt
  )


# ------------------------------------------------------------
# 11. CONVERT CHARACTER VARIABLES TO FACTORS
# ------------------------------------------------------------

classification_data <- classification_data %>%
  mutate(
    across(
      where(is.character),
      as.factor
    )
  )


# ------------------------------------------------------------
# 12. TRAIN / TEST SPLIT
# ------------------------------------------------------------

set.seed(123)

train_index <- createDataPartition(
  classification_data$Price_Category,
  p = 0.80,
  list = FALSE
)

train_data <- classification_data[
  train_index,
]

test_data <- classification_data[
  -train_index,
]


# ------------------------------------------------------------
# 13. CHECK TRAINING AND TESTING DATA
# ------------------------------------------------------------

cat("\nTraining rows:",
    nrow(train_data), "\n")

cat("Testing rows:",
    nrow(test_data), "\n")


cat("\nTraining class distribution:\n")
print(table(train_data$Price_Category))

cat("\nTesting class distribution:\n")
print(table(test_data$Price_Category))


# ------------------------------------------------------------
# 14. SAVE PREPARED DATA
# ------------------------------------------------------------

write.csv(
  train_data,
  "data/train_data.csv",
  row.names = FALSE
)

write.csv(
  test_data,
  "data/test_data.csv",
  row.names = FALSE
)


cat("\nData preparation completed successfully.\n")