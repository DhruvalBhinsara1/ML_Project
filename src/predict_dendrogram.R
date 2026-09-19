# SRC/predict_dendrogram.R
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript SRC/predict_dendrogram.R <linkage> <clusters>")
}

linkage <- trimws(args[1])
clusters <- as.numeric(args[2])

dir.create("static", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)
plot_path <- "static/dendrogram.png"

if (file.exists("SRC/preprocessing.R")) {
  source("SRC/preprocessing.R")
} else if (file.exists("src/preprocessing.R")) {
  source("src/preprocessing.R")
} else {
  source("preprocessing.R")
}

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

method_selected <- linkage_map[[tolower(linkage)]]
if (is.null(method_selected)) {
  method_selected <- "ward.D2"
}

clustering_model <- hclust(distance_matrix, method = method_selected)

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

invisible(dev.off())

# Also save to outputs for standalone usage
file.copy(plot_path, "outputs/dendrogram.png", overwrite = TRUE)

cat("\nRESULT:", plot_path, "\n")
