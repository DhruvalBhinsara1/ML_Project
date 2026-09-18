# src/dendrogram.R

source("src/preprocessing.R")

cat("Loading data for Clustering...\n")
df <- preprocess_data()

# Select features for clustering
features <- c("Size_in_SqFt", "BHK", "Age_of_Property", "Price_in_Lakhs")
df_cluster <- df[, features]

# Scale the data
df_scaled <- scale(df_cluster)

# Sample the data if it's too large, dendrograms get very messy with > 100 points
set.seed(42)
if (nrow(df_scaled) > 100) {
    df_sample <- df_scaled[sample(nrow(df_scaled), 50), ]
} else {
    df_sample <- df_scaled
}

cat("\n=========================================\n")
cat("          DENDROGRAM / CLUSTERING        \n")
cat("=========================================\n")

# Calculate distance
dist_matrix <- dist(df_sample, method = "euclidean")

# Hierarchical clustering
hc_ward <- hclust(dist_matrix, method = "ward.D2")

cat("Hierarchical clustering computed using Ward's method and Euclidean distance.\n")
cat("To plot this in a UI, you would typically generate an image plot.\n")

# In R terminal, you can plot it using:
# plot(hc_ward, main="House Clusters Dendrogram", xlab="", sub="", cex=.9)
# rect.hclust(hc_ward, k=3, border="red")

cat("Clustering algorithm completed successfully.\n")
