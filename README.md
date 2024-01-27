# CaloRadar: Food Image Classifier and Nutrition Information App

## Overview

CaloRadar is an application that combines image classification and nutrition information retrieval to help users identify and learn more about the food items they are about to consume. The project consists of two main components: a deep learning model for image classification trained on a dataset of food images and an integration with the USDA FoodData Central API for retrieving nutritional information.

## Image Classification Model

The image classification model is built using the Keras deep learning library with a base architecture of Xception pre-trained on the ImageNet dataset. The model is trained to classify food items into predefined categories such as "Apple," "Banana," "Bread," "Burger," "Egg," "Fries," "Steak," "Pasta," "Fried Rice," and "Soup." The training data is sourced from Flickr using the FlickrAPI, and the model is fine-tuned to achieve accurate predictions.

## Data Collection

The project includes a data collection script that downloads food images from Flickr for each specified category. The script utilizes the FlickrAPI to search for food-related images, download them, and organize them into directories for training. The dataset includes diverse images of food items to enhance the model's ability to generalize.

## Training and Evaluation

The model is trained using the collected dataset, and its performance is evaluated on a separate set of test images. The evaluation metrics include accuracy, confusion matrix, and class-level accuracy, precision, and recall. These metrics provide insights into how well the model can distinguish between different food categories.

## Nutrition Information Integration

After classifying an image, CaloRadar retrieves nutritional information for the identified food item using the USDA FoodData Central API. The application displays the caloric content and additional nutritional details, offering users insights into the nutritional composition of the food they are about to consume.

## Web Application

CaloRadar is implemented as a web application using the Shiny framework. The user interface allows users to upload an image, classify the food item, view nutritional information, and receive recipe recommendations based on dietary preferences and intolerances. The application integrates shinyjs for interactive features and Shinyalert for displaying allergen alerts.

## Usage

To use CaloRadar, follow these steps:

1. Upload an image of the food item you want to classify.
2. Click the "Classify" button to obtain the predicted food category.
3. View the nutritional information and additional details about the identified food item.
4. Optionally, search for recipe recommendations based on dietary preferences and intolerances.

## Dependencies

CaloRadar relies on several R packages, including `keras`, `tidyverse`, `FlickrAPI`, `shiny`, `shinyjs`, and others. Make sure to install these packages before running the application.

## How to Run

To run the CaloRadar application, open the provided R script and execute the code. The web application will launch, allowing you to interact with the image classifier and nutritional information retrieval system.

## Project Structure

- **Data Collection:** The `data_collection.R` script downloads food images from Flickr for training the image classification model.
- **Image Classifier:** The `image_classifier.R` script defines, trains, and evaluates the deep learning model.
- **Web Application:** The `app.R` script contains the code for the Shiny web application, integrating image classification and nutrition information retrieval.


