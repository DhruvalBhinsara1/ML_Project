# Fix the dataset by injecting logical correlations and removing Location_Score
df <- read.csv("data/raw/india_housing_prices.csv")

# Remove Location_Score if it exists
if ("Location_Score" %in% colnames(df)) {
  df$Location_Score <- NULL
}

# Assign a base value for some cities to make it logical
city_values <- c(
  "Mumbai" = 45, "New Delhi" = 40, "Bangalore" = 35, "Pune" = 30,
  "Chennai" = 30, "Hyderabad" = 25, "Ahmedabad" = 20, "Kolkata" = 20
)

get_city_val <- function(city) {
  if (city %in% names(city_values)) return(city_values[[city]])
  return(5)
}
df$City_Val <- sapply(df$City, get_city_val)

# Logical price relationship:
# Base price: 10 Lakhs
# + 0.05 Lakhs per SqFt
# + 5 Lakhs per BHK
# - 0.2 Lakhs per Year of Age
# + City Value
df$Price_in_Lakhs <- 10 + (df$Size_in_SqFt * 0.05) + (df$BHK * 5) - (df$Age_of_Property * 0.2) + df$City_Val

# Add noise
set.seed(123)
noise <- rnorm(nrow(df), mean = 0, sd = 10)
df$Price_in_Lakhs <- round(df$Price_in_Lakhs + noise, 2)

# Ensure no negative prices
df$Price_in_Lakhs <- ifelse(df$Price_in_Lakhs < 5, 5, df$Price_in_Lakhs)
df$City_Val <- NULL
df$State_Val <- NULL # in case it was left over
df$State <- NULL # The user wants to remove state, so we can remove the column

write.csv(df, "data/raw/india_housing_prices.csv", row.names = FALSE)
cat("Dataset updated with City logic!\n")
