library(modleR)
library(geodata)
library(terra)
library(raster)
library(dplyr)
library(sf)
library(rnaturalearth)        
library(rnaturalearthdata)    


RUN <- 5

# Ler ocorrências de um arquivo .csv
occs <- read.csv(paste0('./DATA_Maxent/Batch_',RUN,'/trainval.csv'))

# Criar sf (Simple Features) das ocorrências, agrupando as colunas de coordenadas
occs_sf <- st_as_sf(occs, coords = c("lon", "lat"), crs = 4326)

# Baixar variáveis bioclimáticas (19 camadas ambientais do WorldClim)
bio_vars <- geodata::worldclim_global(var = "bio", res = 2.5, path = '/path')
bio_stack <- raster::stack(bio_vars)

# Baixar limite político do Reino Unido
uk <- ne_countries(scale = "medium", country = "United Kingdom", returnclass = "sf")

# Transformar variáveis ambientais para formato terra
bio_terra <- terra::rast(bio_stack)

# Reprojetar o limite do Reino Unido para o CRS das variáveis (WGS84)
uk_proj <- st_transform(uk, crs = crs(bio_terra))

# Recortar (crop + mask) as variáveis para o Reino Unido
bio_crop <- terra::crop(bio_terra, vect(uk_proj))
bio_mask <- terra::mask(bio_crop, vect(uk_proj))

# Definir CRS UTM baseado nas ocorrências
mean_lon <- mean(occs$lon)
utm_zone <- floor((mean_lon + 180) / 6) + 1
utm_crs <- paste0("+proj=utm +zone=", utm_zone, " +datum=WGS84 +units=m +no_defs")

# Projetar ocorrências e variáveis para UTM
occs_utm <- st_transform(occs_sf, crs = utm_crs)
bio_utm <- terra::project(bio_mask, utm_crs)

# Converter SpatRaster para RasterStack
bio_utm_raster <- raster::stack()
for (i in 1:terra::nlyr(bio_utm)) {
  bio_utm_raster <- raster::addLayer(bio_utm_raster, raster::raster(bio_utm[[i]]))
}

# Preparar pasta base
output_base_dir <- paste0('modelos_maxent_2/Batch_',RUN,'/')
dir.create(output_base_dir, showWarnings = FALSE)

# Lista de espécies
species_list <- unique(occs$species)


# Loop para cada espécie
for (sp in species_list) {
  message("Rodando para espécie: ", sp)
  
  sp_occurs <- occs_utm %>% 
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
    clean_uni = FALSE,
    clean_nas = TRUE
  )
  message("Finalizado: ", sp)
}


for (sp in species_list) {
  message("Rodando para espécie: ", sp)
  
  sp_occurs <- occs_utm %>% 
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
  message("Finalizado: ", sp)
}


for (sp in species_list) {
  message("Rodando para espécie: ", sp)
  
  sp_occurs <- occs_utm %>% 
    filter(species == sp) %>% 
    st_coordinates() %>% 
    as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  # Gerar modelo final (agregação das partições)
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
  message("Finalizado: ", sp)
}


for (sp in species_list) {
  message("Rodando para espécie: ", sp)
  
  sp_occurs <- occs_utm %>% 
    filter(species == sp) %>% 
    st_coordinates() %>% 
    as.data.frame()
  sp_occurs$species <- sp
  colnames(sp_occurs) <- c("lon", "lat", "species")
  
  ens <- ensemble_model(species_name = sp,
                      occurrences = occs,
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
  message("Finalizado: ", sp)
}
message("Todos os modelos finalizados com ensemble.")




########################################### TEST PREDICTIONS #################################################


library(terra)
library(sf)
library(dplyr)
library(stringr)

# Caminho principal dos modelos
modelos_dir <- paste0('modelos_maxent_2/Batch_',RUN,'/')

# CSV com coordenadas de teste
test_csv <- paste0('./DATA_Maxent/Batch_',RUN,'/test.csv')
test_df <- read.csv(test_csv)

# Converter para sf em WGS84
test_sf <- st_as_sf(test_df, coords = c("lon", "lat"), crs = 4326)

# Lista de espécies (pastas)
species_dirs <- list.dirs(modelos_dir, recursive = FALSE)

# Criar data.frame com coordenadas originais
result_all <- test_df

# Loop por espécie
for (species_path in species_dirs) {
  # Nome da espécie
  species_name <- basename(species_path)
  
  # Caminho do .tif
  tif_file <- file.path(species_path, "present", "final_models", 
                        paste0(species_name, "_maxent_raw_mean.tif"))
  
  if (file.exists(tif_file)) {
    cat("Processando:", species_name, "\n")
    
    # Carrega raster
    model_rast <- rast(tif_file)
    
    # Reprojetar os pontos para o CRS do raster
    test_proj <- st_transform(test_sf, crs = crs(model_rast))
    test_vect <- terra::vect(test_proj)
    
    # Extrair valores
    predicted_values <- terra::extract(model_rast, test_vect)
    predicted_values <- predicted_values[, -1, drop = FALSE]  # Remove ID
    
    # Renomear coluna com nome da espécie
    colnames(predicted_values) <- species_name
    
    # Adicionar ao resultado geral
    result_all[[species_name]] <- predicted_values[[1]]
    
  } else {
    cat("AVISO: Modelo não encontrado para:", species_name, "\n")
  }
}

# Salvar resultado final
write.csv(result_all,  paste0(modelos_dir,'predicoes_todas_as_especies.csv'), row.names = FALSE)

# Últimas 18 colunas: colunas de probabilidade por classe
prob_cols <- tail(colnames(result_all), 18)
class_names <- prob_cols  # nomes das espécies

# Função para calcular top-k
get_topk_flag <- function(probs_row, true_label, k) {
  probs_named <- setNames(as.numeric(probs_row), class_names)
  top_k <- names(sort(probs_named, decreasing = TRUE))[1:min(k, length(probs_named))]
  return(as.integer(true_label %in% top_k))
}

# Calcula top-1, 3, 5, 10
for (k in c(1, 3, 5, 10)) {
  result_all[[paste0("top_", k, "_accuracy")]] <- mapply(
    FUN = function(true_label, probs_row) {
      get_topk_flag(probs_row, true_label, k)
    },
    result_all$truth,
    split(result_all[, prob_cols], seq(nrow(result_all)))
  )
}


write.csv(result_all, paste0(modelos_dir,'predicoes_todas_as_especies_com_acuracia.csv'), row.names = FALSE)


# Vetor com os nomes das colunas de acurácia
accuracy_cols <- c("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "top_10_accuracy")

cat("Acurácias gerais médias:\n")

for (col in accuracy_cols) {
  if (col %in% colnames(result_all)) {
    mean_acc <- mean(result_all[[col]], na.rm = TRUE)
    cat(sprintf("%s: %.2f%%\n", col, mean_acc * 100))
  } else {
    cat(sprintf("Coluna %s não encontrada no data.frame.\n", col))
  }
}



for (col in accuracy_cols) {
  if (col %in% colnames(result_all)) {
    cat(sprintf("\n%s por classe:\n", col))
    result_all %>%
      group_by(species) %>%
      summarise(media = mean(.data[[col]], na.rm = TRUE)) %>%
      arrange(desc(media)) %>%
      mutate(media = sprintf("%.2f%%", media * 100)) %>%
      print(n = Inf)
  }
}


########################################### VALIDATION PREDICTIONS #################################################

library(terra)
library(sf)
library(dplyr)
library(stringr)

RUN <-5

# Caminho principal dos modelos
modelos_dir <- paste0('modelos_maxent_2/Batch_',RUN,'/')

# CSV com coordenadas de teste
test_csv <- paste0('./DATA_Maxent/Batch_',RUN,'/validation.csv')
test_df <- read.csv(test_csv)

# Converter para sf em WGS84
test_sf <- st_as_sf(test_df, coords = c("lon", "lat"), crs = 4326)

# Lista de espécies (pastas)
species_dirs <- list.dirs(modelos_dir, recursive = FALSE)

# Criar data.frame com coordenadas originais
result_all <- test_df

# Loop por espécie
for (species_path in species_dirs) {
  # Nome da espécie
  species_name <- basename(species_path)
  
  # Caminho do .tif
  tif_file <- file.path(species_path, "present", "final_models", 
                        paste0(species_name, "_maxent_raw_mean.tif"))
  
  if (file.exists(tif_file)) {
    cat("Processando:", species_name, "\n")
    
    # Carrega raster
    model_rast <- rast(tif_file)
    
    # Reprojetar os pontos para o CRS do raster
    test_proj <- st_transform(test_sf, crs = crs(model_rast))
    test_vect <- terra::vect(test_proj)
    
    # Extrair valores
    predicted_values <- terra::extract(model_rast, test_vect)
    predicted_values <- predicted_values[, -1, drop = FALSE]  # Remove ID
    
    # Renomear coluna com nome da espécie
    colnames(predicted_values) <- species_name
    
    # Adicionar ao resultado geral
    result_all[[species_name]] <- predicted_values[[1]]
    
  } else {
    cat("AVISO: Modelo não encontrado para:", species_name, "\n")
  }
}

# Salvar resultado final
write.csv(result_all,  paste0(modelos_dir,'predicoes_todas_as_especies_val.csv'), row.names = FALSE)

# Últimas 18 colunas: colunas de probabilidade por classe
prob_cols <- tail(colnames(result_all), 18)
class_names <- prob_cols  # nomes das espécies

# Função para calcular top-k
get_topk_flag <- function(probs_row, true_label, k) {
  probs_named <- setNames(as.numeric(probs_row), class_names)
  top_k <- names(sort(probs_named, decreasing = TRUE))[1:min(k, length(probs_named))]
  return(as.integer(true_label %in% top_k))
}

# Calcula top-1, 3, 5, 10
for (k in c(1, 3, 5, 10)) {
  result_all[[paste0("top_", k, "_accuracy")]] <- mapply(
    FUN = function(true_label, probs_row) {
      get_topk_flag(probs_row, true_label, k)
    },
    result_all$truth,
    split(result_all[, prob_cols], seq(nrow(result_all)))
  )
}


write.csv(result_all, paste0(modelos_dir,'predicoes_todas_as_especies_com_acuracia_val.csv'), row.names = FALSE)


# Vetor com os nomes das colunas de acurácia
accuracy_cols <- c("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "top_10_accuracy")

cat("Acurácias gerais médias:\n")

for (col in accuracy_cols) {
  if (col %in% colnames(result_all)) {
    mean_acc <- mean(result_all[[col]], na.rm = TRUE)
    cat(sprintf("%s: %.2f%%\n", col, mean_acc * 100))
  } else {
    cat(sprintf("Coluna %s não encontrada no data.frame.\n", col))
  }
}



for (col in accuracy_cols) {
  if (col %in% colnames(result_all)) {
    cat(sprintf("\n%s por classe:\n", col))
    result_all %>%
      group_by(species) %>%
      summarise(media = mean(.data[[col]], na.rm = TRUE)) %>%
      arrange(desc(media)) %>%
      mutate(media = sprintf("%.2f%%", media * 100)) %>%
      print(n = Inf)
  }
}





Val:
1:
Acurácias gerais médias:
top_1_accuracy: 29.49%
top_3_accuracy: 66.04%
top_5_accuracy: 81.14%
top_10_accuracy: 93.70%

2:
Acurácias gerais médias:
top_1_accuracy: 28.64%
top_3_accuracy: 70.39%
top_5_accuracy: 79.93%
top_10_accuracy: 93.63%

3:
Acurácias gerais médias:
top_1_accuracy: 33.40%
top_3_accuracy: 70.53%
top_5_accuracy: 80.68%
top_10_accuracy: 93.51%

4:
Acurácias gerais médias:
top_1_accuracy: 28.66%
top_3_accuracy: 68.18%
top_5_accuracy: 80.33%
top_10_accuracy: 93.92%

5:
Acurácias gerais médias:
top_1_accuracy: 25.65%
top_3_accuracy: 65.47%
top_5_accuracy: 79.84%
top_10_accuracy: 93.92%

