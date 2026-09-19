# SRC/predict_knn.R
suppressPackageStartupMessages({
  library(caret)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 8) {
  stop("Usage: Rscript SRC/predict_knn.R <bhk> <size> <year_built> <floor_no> <total_floors> <age> <nearby_schools> <nearby_hospitals> [<city>]")
}

bhk <- as.numeric(args[1])
size <- as.numeric(args[2])
year_built <- as.numeric(args[3])
floor_no <- as.numeric(args[4])
total_floors <- as.numeric(args[5])
age <- as.numeric(args[6])
nearby_schools <- as.numeric(args[7])
nearby_hospitals <- as.numeric(args[8])
city_filter <- if (length(args) >= 9) trimws(args[9]) else "All"

if (any(is.na(c(bhk, size, year_built, floor_no, total_floors, age, nearby_schools, nearby_hospitals)))) {
  stop("All 8 arguments must be valid numeric values.")
}

candidate_paths <- c(
  "models/knn_model.rds",
  "../models/knn_model.rds",
  "SRC/models/knn_model.rds",
  file.path(getwd(), "models", "knn_model.rds")
)

model_path <- candidate_paths[file.exists(candidate_paths)][1]
if (is.na(model_path) || !file.exists(model_path)) {
  stop("KNN model file not found at models/knn_model.rds")
}

knn_bundle <- readRDS(model_path)

input_df <- data.frame(
  BHK = bhk,
  Size_in_SqFt = size,
  Year_Built = year_built,
  Floor_No = floor_no,
  Total_Floors = total_floors,
  Age_of_Property = age,
  Nearby_Schools = nearby_schools,
  Nearby_Hospitals = nearby_hospitals
)

scaled_input <- as.matrix(predict(knn_bundle$preprocessing, input_df))
pred_class <- predict(knn_bundle$model, scaled_input, type = "class")
pred_probs <- predict(knn_bundle$model, scaled_input, type = "prob")

budget_prob <- if ("Budget" %in% colnames(pred_probs)) round(pred_probs[1, "Budget"] * 100, 1) else 0.0
mid_prob <- if ("Mid-Range" %in% colnames(pred_probs)) round(pred_probs[1, "Mid-Range"] * 100, 1) else 0.0
premium_prob <- if ("Premium" %in% colnames(pred_probs)) round(pred_probs[1, "Premium"] * 100, 1) else 0.0
best_k <- if (!is.null(knn_bundle$best_k)) knn_bundle$best_k else 11

cat("\nRESULT:", as.character(pred_class[1]), "|", budget_prob, "|", mid_prob, "|", premium_prob, "|", best_k, "\n")

# --- Retrieve Nearest Real Properties ---
catalog_paths <- c(
  "models/property_catalog.rds",
  "../models/property_catalog.rds",
  "SRC/models/property_catalog.rds",
  file.path(getwd(), "models", "property_catalog.rds")
)
catalog_path <- catalog_paths[file.exists(catalog_paths)][1]

if (!is.na(catalog_path) && file.exists(catalog_path)) {
  catalog <- readRDS(catalog_path)
  X_all <- knn_bundle$model$learn$X

  # Handle city filtering
  is_all_cities <- tolower(city_filter) %in% c("all", "all cities", "nationwide", "any", "")
  candidate_idx <- if (!is_all_cities && city_filter %in% catalog$City) {
    which(catalog$City == city_filter)
  } else {
    seq_len(nrow(X_all))
  }

  if (length(candidate_idx) < 4) {
    candidate_idx <- seq_len(nrow(X_all))
  }

  X_sub <- X_all[candidate_idx, , drop = FALSE]
  diffs <- sweep(X_sub, 2, scaled_input[1, ], "-")
  dists_sq <- rowSums(diffs^2)
  n_pick <- min(6, length(dists_sq))
  top_rel_idx <- order(dists_sq)[1:n_pick]
  top_abs_idx <- candidate_idx[top_rel_idx]

  top_matches <- catalog[top_abs_idx, ]
  top_dists <- sqrt(dists_sq[top_rel_idx])
  match_scores <- round(pmax(10, pmin(99.9, 100 - (top_dists * 16))), 1)

  format_price <- function(lakhs) {
    if (is.na(lakhs)) return("Price on Request")
    if (lakhs >= 100) {
      paste0("₹", sprintf("%.2f", lakhs / 100), " Cr")
    } else {
      paste0("₹", sprintf("%.2f", lakhs), " L")
    }
  }

  formatted_prices <- sapply(top_matches$Price_in_Lakhs, format_price)
  clean_localities <- gsub("_", " ", top_matches$Locality)

  recommended_list <- lapply(seq_len(nrow(top_matches)), function(i) {
    list(
      city = as.character(top_matches$City[i]),
      locality = clean_localities[i],
      property_type = as.character(top_matches$Property_Type[i]),
      bhk = as.integer(top_matches$BHK[i]),
      size = as.numeric(top_matches$Size_in_SqFt[i]),
      price_lakhs = round(as.numeric(top_matches$Price_in_Lakhs[i]), 2),
      formatted_price = formatted_prices[i],
      price_category = as.character(top_matches$Price_Category[i]),
      furnished_status = as.character(top_matches$Furnished_Status[i]),
      floor_no = as.integer(top_matches$Floor_No[i]),
      total_floors = as.integer(top_matches$Total_Floors[i]),
      age = as.numeric(top_matches$Age_of_Property[i]),
      nearby_schools = as.integer(top_matches$Nearby_Schools[i]),
      nearby_hospitals = as.integer(top_matches$Nearby_Hospitals[i]),
      transport = as.character(top_matches$Public_Transport_Accessibility[i]),
      amenities = as.character(top_matches$Amenities[i]),
      match_score = match_scores[i],
      distance = round(top_dists[i], 3)
    )
  })

  if (requireNamespace("jsonlite", quietly = TRUE)) {
    cat("\nPROPERTIES_JSON:", jsonlite::toJSON(recommended_list, auto_unbox = TRUE), "\n")
  }
}
