# ============================================================
# House Price ML Project
# Member 4 - Hierarchical Clustering / Dendrogram
# ============================================================


# ------------------------------------------------------------
# 1. Load housing data
# ------------------------------------------------------------

load_housing_data <- function(file_path) {

  data <- read.csv(
    file_path,
    stringsAsFactors = FALSE
  )

  return(data)
}


# ------------------------------------------------------------
# 2. Prepare data for clustering
# ------------------------------------------------------------

prepare_clustering_data <- function(data) {

  clustering_data <- data[, c(
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals"
  )]

  clustering_data[] <- lapply(
    clustering_data,
    function(x) as.numeric(as.character(x))
  )

  clustering_data <- na.omit(clustering_data)

  return(clustering_data)
}


# ------------------------------------------------------------
# 3. Sample data
# ------------------------------------------------------------

sample_clustering_data <- function(
    data,
    sample_size = 2000) {

  if (nrow(data) > sample_size) {

    set.seed(123)

    data <- data[
      sample(
        seq_len(nrow(data)),
        sample_size
      ),
      ,
      drop = FALSE
    ]
  }

  return(data)
}


# ------------------------------------------------------------
# 4. Scale data
# ------------------------------------------------------------

scale_clustering_data <- function(data) {

  scaled_data <- scale(data)

  return(scaled_data)
}


# ------------------------------------------------------------
# 5. Distance method
# ------------------------------------------------------------

get_distance_method <- function(distance_name) {

  distance_map <- c(
    "Euclidean" = "euclidean",
    "Manhattan" = "manhattan",
    "Maximum" = "maximum",
    "Canberra" = "canberra"
  )

  if (!(distance_name %in% names(distance_map))) {
    stop("Invalid distance method.")
  }

  return(distance_map[[distance_name]])
}


# ------------------------------------------------------------
# 6. Calculate distance matrix
# ------------------------------------------------------------

calculate_distance <- function(
    scaled_data,
    distance_method = "euclidean") {

  distance_matrix <- dist(
    scaled_data,
    method = distance_method
  )

  return(distance_matrix)
}


# ------------------------------------------------------------
# 7. Linkage method
# ------------------------------------------------------------

get_linkage_method <- function(linkage_name) {

  linkage_map <- c(
    "Ward" = "ward.D2",
    "Complete" = "complete",
    "Average" = "average",
    "Single" = "single"
  )

  if (!(linkage_name %in% names(linkage_map))) {
    stop("Invalid linkage method.")
  }

  return(linkage_map[[linkage_name]])
}


# ------------------------------------------------------------
# 8. Perform hierarchical clustering
# ------------------------------------------------------------

perform_hierarchical_clustering <- function(
    distance_matrix,
    linkage_method = "ward.D2") {

  clustering_model <- hclust(
    distance_matrix,
    method = linkage_method
  )

  return(clustering_model)
}


# ------------------------------------------------------------
# 9. Assign clusters
# ------------------------------------------------------------

assign_clusters <- function(
    clustering_model,
    number_of_clusters = 3) {

  if (number_of_clusters < 2) {
    stop("Number of clusters must be at least 2.")
  }

  if (
    number_of_clusters >
    length(clustering_model$order)
  ) {
    stop(
      "Number of clusters cannot exceed ",
      "number of observations."
    )
  }

  clusters <- cutree(
    clustering_model,
    k = number_of_clusters
  )

  return(clusters)
}


# ------------------------------------------------------------
# 10. Create dendrogram
# ------------------------------------------------------------

create_dendrogram <- function(
    clustering_model,
    number_of_clusters = 3) {

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
    k = number_of_clusters
  )
}


# ============================================================
# MAIN PROGRAM
# ============================================================

file_path <- "data/raw/india_housing_prices.csv"

cat("Loading data...\n")

housing_data <- load_housing_data(
  file_path
)

cat(
  "Original rows:",
  nrow(housing_data),
  "\n"
)


cat("Preparing clustering data...\n")

clustering_data <- prepare_clustering_data(
  housing_data
)

cat(
  "Rows after cleaning:",
  nrow(clustering_data),
  "\n"
)

cat(
  "Clustering features:",
  ncol(clustering_data),
  "\n"
)


cat("Sampling data...\n")

clustering_sample <- sample_clustering_data(
  clustering_data,
  2000
)

cat(
  "Sample size:",
  nrow(clustering_sample),
  "\n"
)


cat("Scaling data...\n")

scaled_data <- scale_clustering_data(
  clustering_sample
)


cat("Calculating distance...\n")

distance_method <- get_distance_method(
  "Euclidean"
)

distance_matrix <- calculate_distance(
  scaled_data,
  distance_method
)


cat("Performing clustering...\n")

linkage_method <- get_linkage_method(
  "Ward"
)

clustering_model <-
  perform_hierarchical_clustering(
    distance_matrix,
    linkage_method
  )


cat("Creating clusters...\n")

clusters <- assign_clusters(
  clustering_model,
  3
)


cat("\nCluster sizes:\n")

print(table(clusters))


cat("\nCreating dendrogram...\n")

png(
  "outputs/dendrogram.png",
  width = 1200,
  height = 800
)

create_dendrogram(
  clustering_model,
  3
)

dev.off()

cat(
  "Dendrogram saved to outputs/dendrogram.png\n"
)