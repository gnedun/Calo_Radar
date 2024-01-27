library(shiny)
library(shinydashboard)
library(shinyjs)
library(shinyWidgets)
library(keras)
library(shinyalert)
library(DT)
library(httr)
library(dplyr)


install.packages("reticulate")

# Load your saved model
model <- load_model_hdf5("AI Project/Trained_model_final.h5")

#label_list <- dir("AI Project/train/")
#output_n <- length(label_list)
#save(label_list, file = "label_list.R")
label_list <- c("Apple", "Banana", "Bread", "Burger", "Egg", "Fries", "Steak", "Pasta", "Fried Rice", "Soup")


food_allergens <- list(
  "Bread" = c("Wheat (Gluten)"),
  "Burger" = c("Wheat (Gluten)", "Soy", "Milk or Eggs (depending on the recipe)"),
  "Egg" = c("Eggs"),
  "Fried Rice" = c("Soy", "Possibly Eggs"),
  "Fries" = c("Possibly Potatoes", "Possibly Soy"),
  "Pasta" = c("Wheat (Gluten)"),
  "Soup" = c("Milk", "Soy", "Wheat (depends on the type of soup)")
)

# Function to get data from Spoonacular API
get_spoonacular_data <- function(api_key, query, dietary_option = "None", intolerance_option = "None", num_results = 5) {
  base_url <- "https://api.spoonacular.com/recipes/complexSearch"
  
  # Set up parameters for the request
  params <- list(
    apiKey = api_key,
    query = query,
    addRecipeInformation = TRUE,
    number = num_results
  )
  
  if (dietary_option != "None") {
    params$diet <- dietary_option
  }
  
  if (intolerance_option != "None") {
    params$intolerances <- intolerance_option
  }
  
  tryCatch({
    # Make the request
    response <- GET(url = base_url, query = params)
    stop_for_status(response)  # Check for errors in the response
    
    # Parse the JSON response
    data <- content(response, "text", encoding = "UTF-8")
    data <- jsonlite::fromJSON(data, flatten = TRUE)
    
    # Check if there are any results
    if (!is.null(data$results) && length(data$results) > 0) {
      results <- data$results
      
      # Return the results
      return(results)
    } else {
      # Return an empty list if no results found
      return(list())
    }
    
  }, error = function(err) {
    cat(sprintf("Error making API request: %s\n", conditionMessage(err)))
    # Return an empty list in case of an error
    return(list())
  })
}

# UI
ui <- fluidPage(
  useShinyjs(),
  dashboardPage(
    dashboardHeader(title = "CaloRadar"),
    dashboardSidebar(
      fileInput("file", "Choose an image", accept = c('image/jpeg', 'image/png')),
      actionButton("classifyBtn", "Classify", icon = icon("play")),
      
      # Dropdown for dietary options
      selectInput("dietaryOption", "Available Dietary Options:",
                  choices = c("None", "Gluten Free", "Ketogenic", "Vegetarian",
                              "Lacto-Vegetarian", "Ovo-Vegetarian", "Vegan",
                              "Pescetarian", "Paleo", "Primal", "Low FODMAP", "Whole30"),
                  selected = "None"),
      
      # Dropdown for intolerance options
      selectInput("intoleranceOption", "Available Intolerance Options:",
                  choices = c("None", "Dairy", "Egg", "Gluten", "Grain", "Peanut",
                              "Seafood", "Sesame", "Shellfish", "Soy", "Sulfite",
                              "Tree Nut", "Wheat"),
                  selected = "None"),
      textInput("query", "Enter a food item:", ""),
      actionButton("searchBtn", "Search", icon = icon("search"))
    ),
    dashboardBody(
      box(
        title = "Selected Image",
        imageOutput("selectedImg"),
        height = 300
      ),
      box(
        title = "The food you are about to eat is",
        textOutput("resultText")
      ),
      box(
        title = "Calories",
        textOutput("caloriesInfo")
      ),
      box(
        title = "Additional Information",
        textOutput("additionalInfo")
      ),
      box(
        title = "Recipe Recommendations",
        tableOutput("recipeTable")
      )
    )
  )
)
# Server
server <- function(input, output, session) {
  classified_food_val <- reactiveVal(NULL)
  
  observe({
    if (is.null(input$file)) {
      shinyjs::runjs("$('#classifyBtn').prop('disabled', true);")
    } else {
      shinyjs::runjs("$('#classifyBtn').prop('disabled', false);")
    }
  })
  
  output$selectedImg <- renderImage({
    if (!is.null(input$file)) {
      list(src = input$file$datapath, contentType = input$file$type, width = "100%")
    }
  }, deleteFile = FALSE)
  
  predictions <- eventReactive(input$classifyBtn, {
    req(input$file)
    tryCatch({
      print("Attempting to make predictions...")
      test_image <- image_load(input$file$datapath, target_size = c(224, 224))
      x <- image_to_array(test_image)
      x <- array_reshape(x, c(1, dim(x)))
      x <- x / 255
      pred <- predict(model, x)
      # Create a data frame with class names and corresponding probabilities
      pred_df <- data.frame("Class" = label_list, "Probability" = as.numeric(t(pred)))
      
      # Order the data frame by probability in descending order and select the top 1
      top_class <- pred_df[order(pred_df$Probability, decreasing = TRUE), ][1, ]
      classified_food_val(top_class$Class)
      updateTextInput(session, "query", value = classified_food_val())  # Update the search input
      classified_food_val()
    }, error = function(e) {
      message("Error in predictions: ", e$message)
      return("Error in making predictions. Please check the console for details.")
    })
  })
  
  output$resultText <- renderText({
    req(input$file)
    predictions()
  })
  
  output$additionalInfo <- renderText({
    classified_food <- classified_food_val()
    
    if (classified_food == "Apple") {
      return("Apples are a good source of fiber and vitamin C.")
    } else if (classified_food == "Banana") {
      return("Bananas are rich in potassium and provide a quick energy boost.")
    } else if (classified_food == "Bread") {
      return("Bread is a staple food and a good source of carbohydrates.")
    } else if (classified_food == "Burger") {
      return("Burgers are a popular fast food item often made with a ground meat patty.")
    } else if (classified_food == "Egg") {
      return("Eggs are a rich source of protein and various essential nutrients.")
    } else if (classified_food == "Fried Rice") {
      return("Fried rice is a dish made from cooked rice stir-fried with ingredients like vegetables and meat.")
    } else if (classified_food == "Fries") {
      return("French fries are a popular snack made by frying potatoes.")
    } else if (classified_food == "Pasta") {
      return("Pasta is a type of Italian food typically made from wheat and water.")
    } else if (classified_food == "Soup") {
      return("Soup is a liquid food typically made by boiling ingredients like vegetables, meat, or legumes.")
    } else if (classified_food == "Steak") {
      return("Steak is a meat generally sliced across the muscle fibers, including beefsteak.")
    } else {
      return("Additional information for other foods.")
    }
  })
  
  
  output$caloriesInfo <- renderText({
    req(classified_food_val())
    
    # specify API endpoint URL and parameters
    url <- "https://api.nal.usda.gov/fdc/v1/foods/search"
    params <- list(api_key = "n1ivodEIAZmzjfRczeaehpZgmb7mOvISJYu644u2", query = classified_food_val())
    
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
    
    nutr_df <- calories_data[[1]]
    
    nutr_df <- nutr_df %>%
      select(-nutrientId, -nutrientNumber, -derivationCode, -
               derivationDescription, -derivationId, -foodNutrientSourceId,
             -foodNutrientSourceCode, -foodNutrientSourceDescription,
             -rank, -indentLevel, -foodNutrientId)
    
    # replace missing values with 0
    nutr_df[is.na(nutr_df)] <- 0
    
    # sort the data frame based on percentDailyValue
    nutr_df <- nutr_df[order(-nutr_df$percentDailyValue), ]
    
    # Print the Calories
    energy_value <- nutr_df[nutr_df$nutrientName == 'Energy', 'value'][1]
    
    # Use paste to concatenate the strings
    calories_info <- paste("It contains", energy_value, "calories.")
    
    # Display an alert about common allergens only if the classified food is present in the list
    if (classified_food_val() %in% names(food_allergens)) {
      alert_text <- paste("Common allergens for", classified_food_val(), "include:", toString(food_allergens[[classified_food_val()]]))
      shinyalert(
        title = "Allergen Alert!",
        text = alert_text,
        type = "warning"
      )
    }
    
    return(calories_info)
  })
  
  output$recipeTable <- renderTable({
    req(input$searchBtn)
    
    # API Key for Spoonacular
    api_key <- '207f1050838749e5907eee34c323217f'
    
    # Get dietary and intolerance options
    dietary_option <- input$dietaryOption
    intolerance_option <- input$intoleranceOption
    
    # Get recipe data from Spoonacular API
    recipes <- get_spoonacular_data(api_key, classified_food_val(), dietary_option, intolerance_option, num_results = 5)
    
    # Create a data frame with relevant information
    if (length(recipes) > 0) {
      df <- data.frame(
        Recipe = recipes$title,
        URL = sprintf('<a href="%s" target="_blank">%s</a>', recipes$sourceUrl, "Link")
      )
      return(df)
    } else {
      # Return an empty data frame if no results found
      return(data.frame())
    }
  }, sanitize.text.function = function(x) x)  # Allow HTML in the table
}

# Run the app
shinyApp(ui, server)
