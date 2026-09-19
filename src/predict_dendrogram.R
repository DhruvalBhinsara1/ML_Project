# src/predict_dendrogram.R
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript src/predict_dendrogram.R <linkage> <clusters>")
}

linkage <- args[1]
clusters <- as.numeric(args[2])

dir.create("static", showWarnings = FALSE)
plot_path <- "static/dendrogram.png"

source("src/preprocessing.R")
df <- preprocess_data()

features_clust <- c("Size_in_SqFt", "BHK", "Age_of_Property", "Price_in_Lakhs")
clustering_data <- df[, features_clust]

set.seed(123)
if (nrow(clustering_data) > 2000) {
    clustering_sample <- clustering_data[sample(seq_len(nrow(clustering_data)), 2000), , drop = FALSE]
} else {
    clustering_sample <- clustering_data
}

scaled_data <- scale(clustering_sample)
distance_matrix <- dist(scaled_data, method = "euclidean")

linkage_map <- c(
  "ward" = "ward.D2",
  "complete" = "complete",
  "average" = "average",
  "single" = "single"
)

clustering_model <- hclust(distance_matrix, method = linkage_map[[tolower(linkage)]])

png(plot_path, width = 1200, height = 800)

plot(
  clustering_model,
  main = "House Hierarchical Clustering",
  xlab = "Houses",
  ylab = "Distance",
  sub = "",
  labels = FALSE
)

rect.hclust(
  clustering_model,
  k = clusters
)

dev.off()

cat(plot_path)
