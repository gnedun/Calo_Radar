############################ Load Libraries ###############################
library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
library(tidyverse)
library(FlickrAPI)
library(caret)
library(magick)
library(jpeg)
library(httr) 
library(dplyr)
library(ggplot2)
library(gridExtra)

############################ Specifying food items & directory ##############

# Specify the food items to download
food_items <- c("burger", "banana", "apple", "pasta")

# Specify the directory to save the images
dir.create("AI Project/train/")
local_dir <- "AI Project/train/"

############## Loop through the food items and download the images ############

# Loop through each food item
for (food_item in food_items) {
  # Create a subdirectory for this food item
  sub_dir <- paste(local_dir, "/", food_item, sep = "")
  
  
  dir.create(sub_dir)
  
  # Specify the number of images to download for this food item
  n_images <- 5500
  
  # Specify the number of images to download per request
  per_page <- 500
  
  # Specify the number of requests needed
  n_requests <- ceiling(n_images / per_page)
  
  # Loop through each request
  for (page in 1:n_requests) {
    # Calculate the starting index for this request
    start <- (page - 1) * per_page + 1
    
    # Make the API request for this page
    photos <- getPhotoSearch(api_key = "a78a61870fb226f1aa6e348cd78c075e",
                             tags = food_item,
                             extras = "url_o",
                             img_size = "m",
                             per_page = per_page,
                             page = page,
                             sort = "interestingness-desc")
    
    # Remove columns with NA values in url_o column
    photos <- subset(photos, !is.na(url_m))
    
    # Loop through each row in the dataframe for this request
    for (i in 1:nrow(photos)) {
      # Get the URL for the current photo
      url <- photos$url_m[i]
      
      # Extract the photo ID from the URL
      photo_id <- gsub(".*/", "", url)
      
      # Calculate the index of this photo
      index <- start + i - 1
      
      # Construct the filename by concatenating the index and the photo ID
      filename <- paste(sub_dir, "/", index, "-", photo_id, ".jpg", sep = "")
      
      # Try to download the photo, and skip if there is an error
      tryCatch(
        download.file(url, filename),
        error = function(e) {
          message(sprintf("Error downloading photo %s: %s", photo_id, e$message))
        }
      )
      
      # Stop the loop if we've reached the desired number of images
      if (index >= n_images) {
        break
      }
    }
  }
}

########################## Exploring the data ############################

# set the path to the directory you want to count the files in
path <- "AI Project/train/"
path_test <- "AI Project/test/"

# recursively list all files in the directory and its subdirectories
file_list <- list.files(path, recursive = TRUE, full.names = TRUE)
file_list_test <- list.files(path_test, recursive = TRUE, full.names = TRUE)

# count the number of image files and total files
image_count_train <- file_list %>% 
  str_detect(".jpg|.jpeg|.png|.gif|.bmp") %>% 
  sum()

image_count_test <- file_list_test %>% 
  str_detect(".jpg|.jpeg|.png|.gif|.bmp") %>% 
  sum()

total_images = image_count_train+image_count_test

# print the results
cat(paste0("Number of Training image files: ", image_count_train, "\n"))
cat(paste0("Number of Test image files: ", image_count_test, "\n"))
cat(paste0("Number of Image files: ", total_images, "\n"))


############################ Processing the data ###############################

label_list <- dir("AI Project/train/")
label_list2 <- dir("AI Project/test/")
output_n <- length(label_list)
save(label_list, file="label_list.R")

width <- 224
height<- 224
target_size <- c(width, height)
rgb <- 3 #color channels

path_train <- "AI Project/train/" # modify the path to the training data if necessary
train_data_gen <- image_data_generator(rescale = 1/255, 
                                       validation_split = .4)

train_images <- flow_images_from_directory(path_train,
                                           train_data_gen,
                                           subset = 'training',
                                           target_size = target_size,
                                           class_mode = "categorical",
                                           shuffle = TRUE, # modify to shuffle the images
                                           batch_size = 32, # modify the batch size if necessary
                                           classes = label_list,
                                           seed = 2021)

validation_images <- flow_images_from_directory(path_train,
                                                train_data_gen,
                                                subset = "validation",
                                                target_size = target_size,
                                                class_mode = "categorical",
                                                shuffle = TRUE,
                                                batch_size = 32,
                                                classes = label_list,
                                                seed = 2021)

table(train_images$classes)

############################ Fitting the Model ###############################
  
mod_base <- application_xception(weights = 'imagenet', 
                                 include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base) 

model_function <- function(learning_rate = 0.001, 
                           dropoutrate=0.2, n_dense=1024){
  
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    mod_base %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = n_dense) %>%
    layer_activation("relu") %>%
    layer_dropout(dropoutrate) %>%
    layer_dense(units=output_n, activation="softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = learning_rate),
    metrics = "accuracy"
  )
  
  return(model)
  
}

model <- model_function()
model

model %>% fit(
  train_images,
  class_mode = "categorical",
  classes = label_list,
  epochs = 5, # modify the number of epochs if necessary
  validation_data = validation_images
)

############################ Save the Model ###############################

save_model_hdf5(model, "AI Project/New_Fire_Project-Group6-FlickRImageClassifier.h5")
save_model_hdf5(model, "AI Project/Trained_model_final.h5", overwrite = TRUE)

############################ Accuracy ###############################

model <- load_model_hdf5("AI Project/Trained_model_final.h5")
label_list <- c("Apple", "Banana", "Bread", "Burger", "Egg", "Fries", "Steak", "Pasta", "Rice", "Soup")

path_test <- "AI Project/test/"
test_data_gen <- image_data_generator(rescale = 1/255)
test_images <- flow_images_from_directory(path_test,
                                          test_data_gen,
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = label_list,
                                          shuffle = F,
                                          seed = 2021)

model %>% evaluate_generator(test_images, 
                             steps = test_images$n)

############################ Confusion Matrix ###############################
# Make predictions on test images
pred <- predict(model, test_images)

# Get predicted labels
pred_labels <- label_list[apply(pred, 1, which.max)]

# Get true labels
true_labels <- test_images$classes

# Create confusion matrix
conf_mat <- table(pred_labels, true_labels)
conf_mat



############################ Class level Accuracy, Precision, Recall ###############################

# Compute class-level accuracy
class_acc <- diag(conf_mat) / colSums(conf_mat, na.rm = TRUE) * 100
names(class_acc) <- label_list
cat("Class-level accuracy:\n")
print(paste(round(class_acc, 2), "%", sep = ""))

# Compute precision
precision <- diag(conf_mat) / colSums(conf_mat) * 100
names(precision) <- label_list
cat("Precision:\n")
print(paste(round(precision, 2), "%", sep = ""))

# Compute recall
recall <- diag(conf_mat) / rowSums(conf_mat) * 100
names(recall) <- label_list
cat("Recall:\n")
print(paste(round(recall, 2), "%", sep = ""))


############################ Testing a custom image downloaded from a google search###############################

#PourHouseAmericanBurger.webp


# Load test image
test_image <- image_load("C:/Users/gnedu/Desktop/Images/how-to-fry-an-egg-3-1200.jpg",
                         target_size = target_size)

# Display the image
img <- readJPEG("C:/Users/gnedu/Desktop/Images/how-to-fry-an-egg-3-1200.jpg")
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, ann = FALSE)
rasterImage(img, 0, 0, 1, 1)

# Preprocess image
x <- image_to_array(test_image)
x <- array_reshape(x, c(1, dim(x)))
x <- x/255


# Make predictions and create a data frame
pred <- model %>% predict(x)

Probability = t(pred)



pred_df <- data.frame("Food" = label_list, Probability)
pred_df <- pred_df[order(pred_df$Probability, decreasing = TRUE),][1:10,]
pred_df$Probability <- paste0(format(round(100*as.numeric(pred_df$Probability), 2), nsmall = 2), " %")
pred_df <- pred_df[complete.cases(pred_df), ]
pred_df

# Display results in a table
#library(knitr)
#kable(pred_df, align = "c", row.names = FALSE, col.names = c("", "Probability"))

############################ Adding the nutritional info ###############################

classified_food = label_list[which.max(t(pred))]

# specify API endpoint URL and parameters
url <- "https://api.nal.usda.gov/fdc/v1/foods/search"
params <- list(api_key = "n1ivodEIAZmzjfRczeaehpZgmb7mOvISJYu644u2", query = classified_food)

# make GET request to API endpoint with parameters and API key
response <- GET(url, query = params)

# extract response content as text
response_text <- content(response, "text")

# parse response text as JSON
response_data <- jsonlite::fromJSON(response_text)


# extract relevant data from response object
food_list <- list(response_data$foods)
food_name <- food_list[[1]]$description
calories_data <- food_list[[1]]$foodNutrients

nutr_df<-calories_data[[1]]

nutr_df <- nutr_df %>%
  select(-nutrientId, -nutrientNumber, -derivationCode, -
           derivationDescription, -derivationId, -foodNutrientSourceId,
         -foodNutrientSourceCode, -foodNutrientSourceDescription, 
         -rank, -indentLevel,-foodNutrientId)



# replace missing values with 0
nutr_df[is.na(nutr_df)] <- 0

# replace missing values with 0
nutr_df[is.na(nutr_df)] <- 0

# sort the data frame based on percentDailyValue
nutr_df <- nutr_df[order(-nutr_df$percentDailyValue),]

# print extracted data to console
cat(paste("Food Name: ", food_name[1], "\n"))
cat("Calories Data: \n")
pri++  nt(nutr_df)

# Print the Calories

energy_value <- nutr_df[nutr_df$nutrientName == 'Energy', 'value'][1]
cat("It contains", energy_value, "calories.")

# Set up plot window grid

#par(mfrow = c(1, 5))

# Display the top 5 nutrients based on Daily Requirement according to the FDA


#par(mfrow = c(1, 5))

# Display the top 5 nutrients based on Daily Requirement according to the FDA

#for (i in 1:5) {
  
  # Create a vector for the data to be used in the pie chart
  pie_data <- c(nutr_df[i,"percentDailyValue"], 100-nutr_df[i,"percentDailyValue"])
  
  # Create a vector for the labels to be used in the pie chart
  pie_labels <- c(paste0(nutr_df[i,"nutrientName"], " - ", nutr_df[i,"percentDailyValue"], "%"), 
                  paste0("Other - ", 100-nutr_df[i,"percentDailyValue"], "%"))
  
  # Create the pie chart
  pie(pie_data, labels = pie_labels, col =  c("#00AFBB", "#FC4E07"), cex.main = 0.8)
  
  # Add the title within the plot area
  title(nutr_df[i,"nutrientName"], line = -15.5, cex.main = 1.2)
}

# Add the main title above the plot area
main_title <- "Top 5 nutrients by daily requirement"
mtext(main_title, outer = TRUE, line = -12.5, cex = 1.5)







