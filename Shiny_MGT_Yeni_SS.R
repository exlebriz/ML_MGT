library(shiny)
library(tidyverse)
library(caret)
library(openair)
library(plotrix)
library(DT)
library(plyr)
library(scales)
library(parallel)
library(DescTools)
library(cowplot)
library(Metrics)
library(caret)
library(kernlab)
library(DiagrammeR)
library(Ckmeans.1d.dp)
library(rcompanion)
library(stats)
library(ggplot2)
library(graphics)
library(tidyverse)
library(ggpubr)  ## Grafiksel uygulamalar icin
library(dplyr)   ## PiPe operatoru ve veri manuplasyonu
library(raster)
library(gstat)
library(plyr)
library(randomForest)
library(parallel)
library(gdistance)
library(DescTools)
library(ellipse)
library(MASS)
library(DescTools)
library(gridExtra)
library(cowplot)
library(Metrics)
library(randomForest)
library(sf)
library(ggpubr)
library(xgboost)
library(kernlab)
library(DiagrammeR)
library(Ckmeans.1d.dp)
library(rcompanion)
library(earth)
library(doParallel)
library(caret)
library(tidyverse) # contains dplyr, ggplot2, etc
library(tidymodels) # has some options for ggplot that make plotting model results easier
library(caretEnsemble) # used to create model stacks in caret
library(doParallel) # allows us to train models in parallel
library(dplyr)
library(reshape2)
library(e1071)
library(fastshap)
library(Cubist)
library(ranger)
library(haven)
library(stringr)
library(brnn)
Sys.setlocale()
Sys.setlocale(category = "LC_ALL", locale = "tr_TR.UTF-8")
#rsconnect::deployApp()


if (file.exists("RIMAGE_2.rdata")) {
  load("RIMAGE_2.rdata")
  print(ls())  # Yüklenen nesneleri listeler
} else {
  stop("RIMAGE_2.rdata dosyası bulunamadı!")
}

# Shiny UI
ui <- fluidPage(
  titlePanel("Predicting Gamow-Teller Transition Matrix Elements Using Machine Learning"),
  
  navlistPanel(
    "Navigation",
    
    tabPanel("Inputs & Prediction",
             sidebarLayout(
               sidebarPanel(
                 h3("Input Values"),
                 fluidRow(
                   column(6, numericInput("A1", "A1:", value = NA, min = 19, max = 34, step = 1)),
                   column(6, numericInput("Z1", "Z1:", value = NA, min = 8, max = 15, step = 1))
                 ),
                 fluidRow(
                   column(6, numericInput("A2", "A2:", value = NA, min = 19, max = 34, step = 1)),
                   column(6, numericInput("Z2", "Z2:", value = NA, min = 9, max = 16, step = 1))
                 ),
                 fluidRow(
                   column(6, numericInput("J", "J:", value = NA, min = 0, max = 10, step = 1)),
                   column(6, numericInput("N", "n:", value = NA, min = 1, max = 4, step = 1))
                 ),
                 fluidRow(
                   column(6, numericInput("Time", "Time:", value = NA, min = 0, max = 6, step = 1)),
                   column(6, numericInput("QEN", "QEN:", value = NA, min = 0.211, max = 18.9))
                 ),
                 fluidRow(
                   column(6, numericInput("IBE", "IBE:", value = NA, min = 3.64, max = 5.022)),
                   column(6, numericInput("LOGft", "LOGft:", value = NA, min = 0.006, max = 2.262))
                 ),
                 actionButton("predict", "Predict"),
                 actionButton("reset", "Clear All Inputs"),
                 downloadButton("download_predictions", "Download Predictions"),
                 
                 # Giriş aralık bilgilerini ekleyelim
                 HTML("<div style='background-color:#f9f9f3; padding:10px; border-radius:8px; margin-top:20px; font-size:0.85em;'>
                        <h4>Input Ranges:</h4>
                        <p><strong>A1:</strong> 19 to 34 (Mass number of the parent nucleus)</p>
                        <p><strong>Z1:</strong> 8 to 15 (Atomic number of the parent nucleus)</p>
                        <p><strong>A2:</strong> 19 to 34 (Mass number of the daughter nucleus)</p>
                        <p><strong>Z2:</strong> 9 to 16 (Atomic number of the daughter nucleus)</p>
                        <p><strong>J:</strong> 0 to 10 (Total angular momentum of final state)</p>
                        <p><strong>n:</strong> 1 to 4 (Index for states with the same J)</p>
                        <p><strong>Time:</strong> 0 to 6 (Half-life)</p>
                        <p><strong>QEN:</strong> 0.211 to 18.9 (Decay energy)</p>
                        <p><strong>IBE:</strong> 3.64 to 5.022 (Branching ratio)</p>
                        <p><strong>LOGft:</strong> 0.006 to 2.262 (Logarithm of ft value)</p>
                      </div>")
               ),
               mainPanel(
                 h3("Upload Data"),
                 fileInput("datafile", "Upload Data File (.txt)"),
                 h4("Prediction Results"),
                 tableOutput("predictions")
               )
             )
    ),
    
    tabPanel("Dataset",
             h3("Train and Test Data"),
             DTOutput("train_data_table"),
             DTOutput("test_data_table")
    ),
    
    tabPanel("Scatter Plot",
             h3("Scatter Plot"),
             plotOutput("scatter_plot")
    ),
    
    tabPanel("Taylor Diagram",
             h3("Taylor Diagram"),
             fluidRow(
               column(8, plotOutput("taylor_diagram")),
               column(4, 
                      h4("Model Labels"),
                      tags$ul(
                        tags$li(tags$span(style="color:blue", "\u25CF"), "XGBoost: This study"),
                        tags$li(tags$span(style="color:black", "\u25CF"), "Random Forest (RF): This study"),
                        tags$li(tags$span(style="color:green", "\u25CF"), "Support Vector Regression (SVR): This study"),
                        tags$li(tags$span(style="color:gray", "\u25CF"), "BRNN: This study"),
                        tags$li(tags$span(style="color:red", "\u25CF"), "Cubist: This study"),
                        tags$li(tags$span(style="color:purple", "\u25A0"), "USDB: Kumar et al., 2021"),
                        tags$li(tags$span(style="color:magenta", "\u25A0"), "IMSRG: Kumar et al., 2021"),
                        tags$li(tags$span(style="color:yellow", "\u25A0"), "CCEI: Kumar et al., 2021"),
                        tags$li(tags$span(style="color:orange", "\u25A0"), "CEFT: Kumar et al., 2021")
                      )
               )
             )
    ),
    
    tabPanel("Evaluation Metrics", 
             h3("Cross-Validation Summary and Test Metrics"),
             plotOutput("bxplt_train"),
             DTOutput("test_metrics_table")
    ),
    # Açıklamaları navlistPanel'in altına ekleyelim
    HTML("<div style='background-color:#f9f9f3; padding:10px; border-radius:12px; margin-top:30px; font-size:0.85em;border: 2px solid black;'>
            <p>Input Descriptions:</p>
            <h6><strong>A1:</strong> Mass number of the parent nucleus.</h6>
            <h6><strong>Z1:</strong> Atomic number of the parent nucleus.</h6>
            <h6><strong>A2:</strong> Mass number of the daughter nucleus.</h6>
            <h6><strong>Z2:</strong> Atomic number of the daughter nucleus.</h6>
            <h6><strong>J:</strong> Total angular momentum of the final state of the daughter nucleus.</h6>
            <h6><strong>n:</strong> Index number used to distinguish states with the same J value in order of energy.</h6>
            <h6><strong>Time:</strong> Half-life of the parent nucleus.</h6>
            <h6><strong>QEN:</strong> Decay energy (Q). It represents the amount of energy (in the MeV unit ) released during decay.</h6>
            <h6><strong>IBE:</strong> Branching ratio (Iβ). It represents the percentage of nuclei that decay to a given energy level (in percentage unit).</h6>
            <h6><strong>LOGft:</strong> Logarithm of the ft value. The ft value gives information about the transition probability of beta decay.</h6>
          </div>")
  )
) 


# Server function
server <- function(input, output, session) {
  
  # Reaktif değerleri tanımla
  A1_value <- reactive({ input$A1 })
  Z1_value <- reactive({ input$Z1 })
  A2_value <- reactive({ input$A2 })
  Z2_value <- reactive({ input$Z2 })
  J_value <- reactive({ input$J })
  N_value <- reactive({ input$N })
  Time_value <- reactive({ input$Time })
  QEN_value <- reactive({ input$QEN })
  IBE_value <- reactive({ input$IBE })
  LOGft_value <- reactive({ input$LOGft })
  
  # Tahmin sonuçlarını depolamak için reactiveVal kullanın
  predictions <- reactiveVal(NULL)
  
  # Giriş kontrolleri
  observe({
    # Kontrolleri her bir girdi için ayrı ayrı tanımla
    if (!is.na(A1_value()) && (A1_value() %% 1 != 0 || A1_value() < 19 || A1_value() > 34)) {
      showNotification("A1 should only be an integer between 19-34", type = "error")
      updateNumericInput(session, "A1", value = 26)
    }
    
    if (!is.na(Z1_value()) && (Z1_value() %% 1 != 0 || Z1_value() < 8 || Z1_value() > 15)) {
      showNotification("Z1 should only be an integer between 8-15", type = "error")
      updateNumericInput(session, "Z1", value = 11)
    }
    
    if (!is.na(A2_value()) && (A2_value() %% 1 != 0 || A2_value() < 19 || A2_value() > 34)) {
      showNotification("A2 should only be an integer between 19-34", type = "error")
      updateNumericInput(session, "A2", value = 26)
    }
    
    if (!is.na(Z2_value()) && (Z2_value() %% 1 != 0 || Z2_value() < 9 || Z2_value() > 16)) {
      showNotification("Z2 should only be an integer between 9-16", type = "error")
      updateNumericInput(session, "Z2", value = 11)
    }
    
    if (!is.na(J_value()) && (J_value() %% 1 != 0 || J_value() < 0 || J_value() > 10)) {
      showNotification("J should only be an integer between 0-10", type = "error")
      updateNumericInput(session, "J", value = 0)
    }
    
    if (!is.na(N_value()) && (N_value() %% 1 != 0 || N_value() < 1 || N_value() > 4)) {
      showNotification("N should only be an integer between 1-4", type = "error")
      updateNumericInput(session, "N", value = 1)
    }
    
    if (!is.na(Time_value()) && (Time_value() %% 1 != 0 || Time_value() < 0 || Time_value() > 6)) {
      showNotification("Time should only be an integer between 0-6", type = "error")
      updateNumericInput(session, "Time", value = 0)
    }
    
    if (!is.na(QEN_value()) && (QEN_value() < 0.211 || QEN_value() > 18.9)) {
      showNotification("QEN should be in the range 0.211 - 18.9", type = "error")
      updateNumericInput(session, "QEN", value = 0.211)
    }
    
    if (!is.na(IBE_value()) && (IBE_value() < 3.64 || IBE_value() > 5.022)) {
      showNotification("IBE should be in the range 3.64 - 5.022", type = "error")
      updateNumericInput(session, "IBE", value = 3.64)
    }
    
    if (!is.na(LOGft_value()) && (LOGft_value() < 0.006 || LOGft_value() > 2.262)) {
      showNotification("LOGft should be in the range 0.006 - 2.262", type = "error")
      updateNumericInput(session, "LOGft", value = 0.006)
    }
  })
  
  # Tahmin işlemi
  observeEvent(input$predict, {
    showNotification("Prediction started...", type = "message")
    input_data <- NULL
    
    # Girişlerin doğruluğunu kontrol et
    if (!is.na(A1_value()) & !is.na(Z1_value()) & !is.na(A2_value()) & !is.na(Z2_value())) {
      input_data <- data.frame(
        A1 = A1_value(), Z1 = Z1_value(),
        A2 = A2_value(), Z2 = Z2_value(),
        J = J_value(), N = N_value(),
        Time = Time_value(), QEN = QEN_value(),
        IBE = IBE_value(), LOGft = LOGft_value()
      )
      showNotification("Manual input data collected successfully.", type = "message")
    } else if (!is.null(input$datafile)) {
      # Dosya yüklenmişse
      data <- tryCatch(read.delim(input$datafile$datapath), error = function(e) NULL)
      if (is.null(data)) {
        showNotification("File loading error.", type = "error")
        return()
      }
      # Veri sütunlarını kontrol et
      required_columns <- c("A1", "Z1", "A2", "Z2", "J", "N", "Time", "QEN", "IBE", "LOGft")
      if (all(required_columns %in% names(data))) {
        input_data <- data[, required_columns]
        showNotification("File input data collected successfully.", type = "message")
      } else {
        showNotification("Uploaded data does not contain required columns.", type = "error")
        return()
      }
    } else {
      showNotification("Please provide input values or upload a data file.", type = "error")
      return()
    }
    
    # İşleme devam
    if (!is.null(input_data)) {
      transformed_data <- tryCatch({
        predict(prepro_DF_model, input_data)
      }, error = function(e) {
        showNotification("Data transformation error.", type = "error")
        NULL
      })
      
      if (!is.null(transformed_data)) {
        predictions_df <- tryCatch({
          data.frame(
            RF = predict(MOD_DF_model$rf, newdata = transformed_data),
            Cubist = predict(MOD_DF_model$cubist, newdata = transformed_data),
            XGBoost = predict(MOD_DF_model$xgblinear, newdata = transformed_data),
            SVM = predict(MOD_DF_model$svmRadial, newdata = transformed_data),
            BRNN = predict(MOD_DF_model$brnn, newdata = transformed_data)
          )
        }, error = function(e) {
          showNotification("Prediction error.", type = "error")
          NULL
        })
        
        if (!is.null(predictions_df)) {
          showNotification("Prediction successful.", type = "message")
          # Tahmin sonuçlarını reactiveVal içine kaydedin
          predictions(predictions_df)
        }
      }
    }
  })
  
  # Tahmin sonuçlarını gösteren output tanımı
  output$predictions <- renderTable({
    req(predictions())
    round(predictions(), 3)
  })
  
  # Tahmin sonuçlarını indirme işlevi
  output$download_predictions <- downloadHandler(
    filename = function() { "predictions.csv" },
    content = function(file) {
      write.csv(predictions(), file, row.names = FALSE)
    }
  )
  
  # Clear all inputs when "reset" button is clicked
  observeEvent(input$reset, {
    updateNumericInput(session, "A1", value = NA)
    updateNumericInput(session, "Z1", value = NA)
    updateNumericInput(session, "A2", value = NA)
    updateNumericInput(session, "Z2", value = NA)
    updateNumericInput(session, "J", value = NA)
    updateNumericInput(session, "N", value = NA)
    updateNumericInput(session, "Time", value = NA)
    updateNumericInput(session, "QEN", value = NA)
    updateNumericInput(session, "IBE", value = NA)
    updateNumericInput(session, "LOGft", value = NA)
    
    # Kullanıcıya girişlerin temizlendiğine dair bildirim göster
    showNotification("All input fields have been cleared.", type = "message")
  })
  # Eğitim ve test veri setleri
  output$train_data_table <- renderDT({
    datatable(Train_DF_model, options = list(pageLength = 5))
  })
  
  output$test_data_table <- renderDT({
    datatable(Test_DF_model, options = list(pageLength = 5))
  })
  # Scatter Plot
  output$scatter_plot <- renderPlot({
    scatterPlot(TD_DF_model,
                x = "Gercek",
                y = "Tahmin",
                z = "Hata",
                type = c("Model"),
                col = "jet",
                smooth = FALSE,
                linear = TRUE,
                key.footer = "Error",
                xlim = c(0, 1.5),
                ylim = c(0, 1.5),
                xbin = 5,
                statistic = "mean",
                xlab = "Experimental M(GT)",
                ylab = "Theoretical M(GT)",
                main = "Testing data cross-validations")
  })
  
  # Taylor Diagram
  output$taylor_diagram <- renderPlot({
    par(pty = "s")
    if (!is.null(dfi_DF_model$observed) && !is.null(dfi_DF_model$XGBoost)) {
      oldpar <- taylor.diagram(dfi_DF_model$observed, dfi_DF_model$XGBoost, col = "blue", pcex = 2, pch = 16)
      taylor.diagram(dfi_DF_model$observed, dfi_DF_model$RF, add = TRUE, col = "black", pcex = 2, pch = 19)
      taylor.diagram(dfi_DF_model$observed, dfi_DF_model$SVR, add = TRUE, col = "green", pcex = 2, pch = 16)
      taylor.diagram(dfi_DF_model$observed, dfi_DF_model$BRNN, add = TRUE, col = "gray", pcex = 2, pch = 16)
      taylor.diagram(dfi_DF_model$observed, dfi_DF_model$Cubist, add = TRUE, col = "red", pcex = 2, pch = 16)
      
      taylor.diagram(dfi_DF_model$observed, KAR_DF_test$USDB, add = TRUE, col = "purple", pcex = 2, pch = 15)
      taylor.diagram(dfi_DF_model$observed, KAR_DF_test$IMSRG, add = TRUE, col = "magenta", pcex = 2, pch = 15)
      taylor.diagram(dfi_DF_model$observed, KAR_DF_test$CCEI, add = TRUE, col = "yellow", pcex = 2, pch = 15)
      taylor.diagram(dfi_DF_model$observed, KAR_DF_test$CEFT, add = TRUE, col = "orange", pcex = 2, pch = 15)
    } else {
      print("Taylor diagram data is missing or incomplete.")
    }
  })
  
  # Cross-validation boxplot for training data
  output$bxplt_train <- renderPlot({
    print(bxplt_train)
  })
  
  # Test metrics table sorted by RMSE
  output$test_metrics_table <- renderDT({
    observed <- dfi_DF_model$observed
    test_metrics <- data.frame(
      Model = rownames(RMSE_Test_DF_model),
      RMSE = RMSE_Test_DF_model$RMSE,
      MAE = RMSE_Test_DF_model$MAE,
      R2 = round(sapply(rownames(RMSE_Test_DF_model), function(model) {
        predicted <- dfi_DF_model[[model]]
        cor(observed, predicted, use = "complete.obs")^2
      }),3)
    )
    
    test_metrics <- test_metrics %>% arrange(RMSE)
    
    datatable(test_metrics, options = list(pageLength = 5), 
              caption = "Test Metrics for Each Model (sorted by RMSE)")
  })
}
# Run the application
shinyApp(ui = ui, server = server)
