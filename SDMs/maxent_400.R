library(modleR)
library(geodata)
library(terra)
library(raster)
library(dplyr)
library(sf)
library(rnaturalearth)        
library(rnaturalearthdata)    


RUN <- 5

# Read occurrences from a CSV file
occurrences <- read.csv(paste0('./DATA_Maxent/Batch_',RUN,'/trainval.csv'))

# Create sf (Simple Features) from occurrences, using coordinate columns
occurrences_sf <- st_as_sf(occurrences, coords = c("lon", "lat"), crs = 4326)

# Download bioclimatic variables (19 WorldClim layers)
bio_vars <- geodata::worldclim_global(var = "bio", res = 2.5, path = '/path')
bio_stack <- raster::stack(bio_vars)

# Download political boundary of the United Kingdom
uk <- ne_countries(scale = "medium", country = "United Kingdom", returnclass = "sf")

# Convert environmental variables to terra format
bio_terra <- terra::rast(bio_stack)

# Reproject UK boundary to the CRS of the variables (WGS84)
uk_proj <- st_transform(uk, crs = crs(bio_terra))

# Crop and mask variables to the United Kingdom
bio_crop <- terra::crop(bio_terra, vect(uk_proj))
bio_mask <- terra::mask(bio_crop, vect(uk_proj))

# Define UTM CRS based on occurrence longitudes
mean_lon <- mean(occurrences$lon)
utm_zone <- floor((mean_lon + 180) / 6) + 1
utm_crs <- paste0("+proj=utm +zone=", utm_zone, " +datum=WGS84 +units=m +no_defs")

# Project occurrences and variables to UTM
occurrences_utm <- st_transform(occurrences_sf, crs = utm_crs)
bio_utm <- terra::project(bio_mask, utm_crs)

# Convert SpatRaster to RasterStack
bio_utm_raster <- raster::stack()
for (i in 1:terra::nlyr(bio_utm)) {
  bio_utm_raster <- raster::addLayer(bio_utm_raster, raster::raster(bio_utm[[i]]))
}

# Prepare base output directory
output_base_dir <- paste0('modelos_maxent/Batch_',RUN,'/')
dir.create(output_base_dir, showWarnings = FALSE)

# Species list
species_list <- unique(occurrences$species)


# Loop per species to set up SDM data
for (sp in species_list) {
  message("Running for species: ", sp)
  
  sp_occurs <- occurrences_utm %>% 
    filter(species == sp) %>% 
    st_coordinates() %>% 
    as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  sdmdata_1sp <- setup_sdmdata(
    species_name = sp,
    occurrences = sp_occurs,
    predictors = bio_utm_raster,
    models_dir = output_base_dir,
    partition_type = "crossvalidation",
    cv_partitions = 5,
    cv_n = 1,
    n_back = 400,
    seed = 512,
    buffer_type = "mean",
    png_sdmdata = TRUE,
    clean_dupl = TRUE,
    clean_uni = TRUE,
    clean_nas = TRUE
  )
  message("Finished: ", sp)
}


for (sp in species_list) {
  message("Running for species: ", sp)
  
  sp_occurs <- occurrences_utm %>% 
    filter(species == sp) %>% 
    st_coordinates() %>% 
    as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  sp_maxnet <- do_any(
    species_name = sp,
    models_dir = output_base_dir,
    predictors = bio_utm_raster,
    algorithm = c("maxent"),
    png_partitions = TRUE,
    write_bin_cut = FALSE,
    equalize = TRUE,
    write_rda = TRUE
  )
  message("Finished: ", sp)
}


for (sp in species_list) {
  message("Running for species: ", sp)
  
  sp_occurs <- occurrences_utm %>% 
    filter(species == sp) %>% 
    st_coordinates() %>% 
    as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  # Generate final model (aggregate partitions)
  final_model(
    species_name = sp,
    models_dir = output_base_dir,
    algorithms = c("maxent"),
    which_models = c("raw_mean"),
    consensus_level = 0.5,
    uncertainty = TRUE,
    overwrite = TRUE,
    write_rds = TRUE
  )
  message("Finished: ", sp)
}


for (sp in species_list) {
  message("Running for species: ", sp)
  
  sp_occurs <- occurrences_utm %>% 
    filter(species == sp) %>% 
    st_coordinates() %>% 
    as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  ens <- ensemble_model(species_name = sp,
                      occurrences = occurrences,
                      performance_metric = "pROC",
                      which_ensemble = c("average",
                                         "best",
                                         "frequency",
                                         "weighted_average",
                                         "median",
                                         "pca",
                                         "consensus"),
                      consensus_level = 0.5,
                      which_final = "raw_mean",
                      models_dir = output_base_dir,
                      overwrite = TRUE
  )                    
  message("Finished: ", sp)
}
message("All models finished with ensemble.")


########################################### TEST PREDICTIONS #################################################


library(terra)
library(sf)
library(dplyr)
library(stringr)

# Main models directory
models_dir <- paste0('modelos_maxent/Batch_',RUN,'/')

# CSV with test coordinates
test_csv <- paste0('./DATA_Maxent/Batch_',RUN,'/test.csv')
test_df <- read.csv(test_csv)

# Convert to sf in WGS84
test_sf <- st_as_sf(test_df, coords = c("lon", "lat"), crs = 4326)

# List of species directories
species_dirs <- list.dirs(models_dir, recursive = FALSE)

# Create data.frame with original coordinates
result_all <- test_df

# Loop per species directory
for (species_path in species_dirs) {
  # species name
  species_name <- basename(species_path)
  
  # path to .tif
  tif_file <- file.path(species_path, "present", "final_models", 
                        paste0(species_name, "_maxent_raw_mean.tif"))
  
  if (file.exists(tif_file)) {
    cat("Processing:", species_name, "\n")
    
    # load raster
    model_rast <- rast(tif_file)
    
    # Reproject points to raster CRS
    test_proj <- st_transform(test_sf, crs = crs(model_rast))
    test_vect <- terra::vect(test_proj)
    
    # Extract values
    predicted_values <- terra::extract(model_rast, test_vect)
    predicted_values <- predicted_values[, -1, drop = FALSE]  # Remove ID
    
    # Rename column with species name
    colnames(predicted_values) <- species_name
    
    # Add to overall result
    result_all[[species_name]] <- predicted_values[[1]]
    
  } else {
    cat("WARNING: Model not found for:", species_name, "\n")
  }
}

# Save final result
write.csv(result_all, paste0(models_dir,'predictions_all_species.csv'), row.names = FALSE)

# Last 18 columns: probability columns per class
prob_cols <- tail(colnames(result_all), 18)
class_names <- prob_cols  # species names

# Function to compute top-k flag
get_topk_flag <- function(probs_row, true_label, k) {
  probs_named <- setNames(as.numeric(probs_row), class_names)
  top_k <- names(sort(probs_named, decreasing = TRUE))[1:min(k, length(probs_named))]
  return(as.integer(true_label %in% top_k))
}

# Compute top-1, 3, 5, 10
for (k in c(1, 3, 5, 10)) {
  result_all[[paste0("top_", k, "_accuracy")]] <- mapply(
    FUN = function(true_label, probs_row) {
      get_topk_flag(probs_row, true_label, k)
    },
    result_all$truth,
    split(result_all[, prob_cols], seq(nrow(result_all)))
  )
}


write.csv(result_all, paste0(models_dir,'predictions_all_species_with_accuracy.csv'), row.names = FALSE)


# Vector with accuracy column names
accuracy_cols <- c("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "top_10_accuracy")

cat("Average overall accuracies:\n")

for (col in accuracy_cols) {
  if (col %in% colnames(result_all)) {
    mean_acc <- mean(result_all[[col]], na.rm = TRUE)
    cat(sprintf("%s: %.2f%%\n", col, mean_acc * 100))
  } else {
    cat(sprintf("Column %s not found in the data.frame.\n", col))
  }
}


for (col in accuracy_cols) {
  if (col %in% colnames(result_all)) {
    cat(sprintf("\n%s by class:\n", col))
    result_all %>%
      group_by(species) %>%
      summarise(mean_acc = mean(.data[[col]], na.rm = TRUE)) %>%
      arrange(desc(mean_acc)) %>%
      mutate(mean_acc = sprintf("%.2f%%", mean_acc * 100)) %>%
      print(n = Inf)
  }
}


########################################### VALIDATION PREDICTIONS #################################################

library(terra)
library(sf)
library(dplyr)
library(stringr)

RUN <-5

# Main models directory
models_dir <- paste0('modelos_maxent/Batch_',RUN,'/')

# CSV with validation coordinates
test_csv <- paste0('./DATA_Maxent/Batch_',RUN,'/validation.csv')
test_df <- read.csv(test_csv)

# Convert to sf in WGS84
test_sf <- st_as_sf(test_df, coords = c("lon", "lat"), crs = 4326)

# List of species directories
species_dirs <- list.dirs(models_dir, recursive = FALSE)

# Create data.frame with original coordinates
result_all <- test_df

# Loop per species directory
for (species_path in species_dirs) {
  # species name
  species_name <- basename(species_path)
  
  # path to .tif
  tif_file <- file.path(species_path, "present", "final_models", 
                        paste0(species_name, "_maxent_raw_mean.tif"))
  
  if (file.exists(tif_file)) {
    cat("Processing:", species_name, "\n")
    
    # load raster
    model_rast <- rast(tif_file)
    
    # Reproject points to raster CRS
    test_proj <- st_transform(test_sf, crs = crs(model_rast))
    test_vect <- terra::vect(test_proj)
    
    # Extract values
    predicted_values <- terra::extract(model_rast, test_vect)
    predicted_values <- predicted_values[, -1, drop = FALSE]  # Remove ID
    
    # Rename column with species name
    colnames(predicted_values) <- species_name
    
    # Add to overall result
    result_all[[species_name]] <- predicted_values[[1]]
    
  } else {
    cat("WARNING: Model not found for:", species_name, "\n")
  }
}

# Save final result
write.csv(result_all,  paste0(models_dir,'predictions_all_species_val.csv'), row.names = FALSE)

# Last 18 columns: probability columns per class
prob_cols <- tail(colnames(result_all), 18)
class_names <- prob_cols  # species names

# Function to compute top-k flag
get_topk_flag <- function(probs_row, true_label, k) {
  probs_named <- setNames(as.numeric(probs_row), class_names)
  top_k <- names(sort(probs_named, decreasing = TRUE))[1:min(k, length(probs_named))]
  return(as.integer(true_label %in% top_k))
}

# Compute top-1, 3, 5, 10
for (k in c(1, 3, 5, 10)) {
  result_all[[paste0("top_", k, "_accuracy")]] <- mapply(
    FUN = function(true_label, probs_row) {
      get_topk_flag(probs_row, true_label, k)
    },
    result_all$truth,
    split(result_all[, prob_cols], seq(nrow(result_all)))
  )
}


write.csv(result_all, paste0(models_dir,'predictions_all_species_with_accuracy_val.csv'), row.names = FALSE)


# Vector with accuracy column names
accuracy_cols <- c("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "top_10_accuracy")

cat("Average overall accuracies:\n")

for (col in accuracy_cols) {
  if (col %in% colnames(result_all)) {
    mean_acc <- mean(result_all[[col]], na.rm = TRUE)
    cat(sprintf("%s: %.2f%%\n", col, mean_acc * 100))
  } else {
    cat(sprintf("Column %s not found in the data.frame.\n", col))
  }
}


for (col in accuracy_cols) {
  if (col %in% colnames(result_all)) {
    cat(sprintf("\n%s by class:\n", col))
    result_all %>%
      group_by(species) %>%
      summarise(mean_acc = mean(.data[[col]], na.rm = TRUE)) %>%
      arrange(desc(mean_acc)) %>%
      mutate(mean_acc = sprintf("%.2f%%", mean_acc * 100)) %>%
      print(n = Inf)
  }
}

