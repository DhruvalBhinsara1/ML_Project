# src/preprocessing.R

preprocess_data <- function(filepath = "data/raw/india_housing_prices.csv") {
  # Load the data
  df <- read.csv(filepath)
  
  # Select only the relevant features we will use for modeling
  # Based on the user requirement and available columns:
  selected_columns <- c(
    "Size_in_SqFt", 
    "BHK", 
    "Age_of_Property", 
    "City",
    "Price_in_Lakhs"
  )
  
  df_clean <- df[, selected_columns]
  
  # Handle any missing values
  df_clean <- na.omit(df_clean)
  
  # Create a category column for the Classification model
  # Budget: bottom 33%, Mid-Range: middle 33%, Premium: top 33%
  quantiles <- quantile(df_clean$Price_in_Lakhs, probs = c(0.33, 0.66), na.rm = TRUE)
  
  categorize_price <- function(p) {
    if (p <= quantiles[1]) {
      return('Budget')
    } else if (p <= quantiles[2]) {
      return('Mid-Range')
    } else {
      return('Premium')
    }
  }
  
  df_clean$Category <- sapply(df_clean$Price_in_Lakhs, categorize_price)
  df_clean$Category <- as.factor(df_clean$Category)
  
  return(df_clean)
}

# If run directly, test the function
if (!interactive()) {
  df <- preprocess_data()
  print(head(df))
  cat("Preprocessing complete. Data shape: ", dim(df), "\n")
}
