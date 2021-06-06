rm(list=ls())
options(scipen=999)

keras::use_python("/home/msmith/anaconda3/bin/python3.6", required = T)

############### Options for Paralell Computing #################
public_ip <- ipify::get_ip()
ssh_private_key_file <- "/home/msmith/pemkey/AWS-RStudio-Key.pem"
parallel:::sinkWorkerOutput

# cl <- makeClusterPSOCK(
#   
#   # Public IP number of EC2 instance
#   workers = public_ip,
#   
#   # User name (always 'ubuntu')
#   user = "ubuntu",
#   
#   # Use private SSH key registered with AWS
#   rshopts = c(
#     "-o", "StrictHostKeyChecking=no",
#     "-o", "IdentitiesOnly=yes",
#     "-i", ssh_private_key_file
#   ),
#   
#   # Set up .libPaths() for the 'ubuntu' user and
#   # install furrr
#   rscript_args = c(
#     "-e", shQuote("local({p <- Sys.getenv('R_LIBS_USER'); dir.create(p, recursive = TRUE, showWarnings = FALSE); .libPaths(p)})"),
#     "-e", shQuote("install.packages('furrr')")
#   ),
#   
#   # Switch this to TRUE to see the code that is run on the workers without
#   # making the connection
#   dryrun = FALSE
# )

#plan(list(tweak(future::cluster, workers = cl), multiprocess))
########### END Options for Paralell Computing #################

library(future.apply)
library(furrr)
library(purrr)
library(dplyr)
library(tidyr)
library(xgboost)
library(xgboostExplainer)
library(Matrix)
library(caret)
library(pROC)
library(PRROC)
library(tidyquant)
library(drlib)
library(keras)
library(tensorflow)
#library(randomForest) # Note: randomForest causes conflicts with ggplot/tidyquant "margin" function
library(lightgbm)
library(e1071)
library(stargazer)
library(gtable)
library(cowplot)
library(grid)
# library(patchwork)
# library(ggtext)       

setwd("/home/msmith/chapter_1")

########################## Read in the data #####################

# files <- list.files("/home/msmith/chapter_1/data", recursive = FALSE, pattern = ".csv")
# files <- paste("/home/msmith/chapter_1/data", files, sep = "/")

files <- list.files("/home/msmith/chapter_1/data/new_ratios", recursive = FALSE, pattern = ".csv")
files <- paste("/home/msmith/chapter_1/data/new_ratios", files, sep = "/")

readdata <- function(fn){
  dt_temp <- read.csv(file = fn, stringsAsFactors = FALSE)
  return(dt_temp)
}

data <- future_lapply(files, readdata)


#################################################################
set.seed(1234)

full_data <- data %>% 
  future_map(., ~as_tibble(.) %>%
               rename(ID = X) %>%
               dplyr::select(-c(EBIT.Capital, EBIT.FinExp, SALES.FA, EBITDA.SALES)) %>%  # vars from stepwise, lasso, ridge, elastic net to remove
               sample_n(nrow(.))
             )

train <- full_data %>% 
  future_map(., ~as_tibble(.) %>% 
               sample_frac(0.75))

test <- full_data %>% 
  future_map2(.x = ., 
              .y = train, 
              .f = ~anti_join(.x, .y, by = "ID"))

#################################################################

X_train <- train %>% 
  future_map(., ~dplyr::select(., -ID, -Postcode, -City, -Latitude, -Longitude, -Region.in.country, 
                               -Major.sectors, -NACE.Rev..2.main.section, -BvD.ID.number, -status) %>% 
               as.matrix() %>% 
               Matrix(., sparse = TRUE)
  )

Y_train <- train %>% 
  future_map(., ~dplyr::select(., status) %>% 
               as.matrix() %>% 
               Matrix(., sparse = TRUE)
  )

X_test <- test %>% 
  future_map(., ~dplyr::select(., -ID, -Postcode, -City, -Latitude, -Longitude, -Region.in.country, 
                               -Major.sectors, -NACE.Rev..2.main.section, -BvD.ID.number, -status) %>% 
               as.matrix() %>% 
               Matrix(., sparse = TRUE)
  )

Y_test <- test %>% 
  future_map(., ~dplyr::select(., status) %>% 
               as.matrix() %>% 
               Matrix(., sparse = TRUE)
  )



#################################################################
#################################################################
#################################################################
#################### Multi ML model #############################
#################################################################
#################################################################
#################################################################
# Scaled and Outlier handled data:

remove_outliers <- function(x, na.rm = TRUE, ...) {
  DescTools::Trim(x, trim = 0.1, na.rm = FALSE)
  qnt <- quantile(x, probs=c(0.1, 0.9), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

scale2 <- function(x, na.rm = TRUE) (x - mean(x, na.rm = na.rm)) / sd(x, na.rm)
#scale2 <- function(x, na.rm = TRUE) (x - min(x, na.rm = na.rm)) / max(x, na.rm = na.rm) - min(x, na.rm = na.rm)
sapply(data[[1]], function(x) sum(is.na(x)))

###################
theme_set(theme_tq() + 
            theme(panel.background=element_rect(colour="grey40", fill=NA)))


#################################################################
################# Neural Network Models ##########################
#################################################################

train_scaled <- train %>%
  map(., 
      ~mutate_at(., vars(11:ncol(.)), funs(scale2(remove_outliers(.)))) %>% 
        mutate_if(is.integer, as.numeric) %>% 
        drop_na(.)
  )

test_scaled <- test %>% 
  map(., 
      ~mutate_at(., vars(11:ncol(.)), funs(scale2(remove_outliers(.)))) %>% 
        mutate_if(is.integer, as.numeric) %>% 
        drop_na(.)
  )

X_train_scaled <- train_scaled %>%
  map(., ~dplyr::select(., -ID, -Postcode, -City, -Latitude, -Longitude, -Region.in.country, 
                 -Major.sectors, -NACE.Rev..2.main.section, -BvD.ID.number, -status) %>% 
        as.matrix())

Y_train_scaled_cat <- train_scaled %>%
  map(., ~pull(., status) %>% 
        keras::to_categorical(2))

X_test_scaled <- test_scaled %>%
  map(., ~select(., -ID, -Postcode, -City, -Latitude, -Longitude, -Region.in.country, 
                 -Major.sectors, -NACE.Rev..2.main.section, -BvD.ID.number, -status) %>% 
        as.matrix())

Y_test_scaled_cat <- test_scaled %>%
  map(., ~pull(., status) %>% 
        to_categorical(2))
#################################################################
map(train_scaled, ~dim(.x))
map(test_scaled, ~dim(.x))

map(train, ~dim(.x))
map(test, ~dim(.x))

map(train, ~select(.x, status) %>% 
      group_by(status) %>% 
      count()
)
map(test, ~select(.x, status) %>% 
      group_by(status) %>% 
      count()
)

map(train_scaled, ~select(.x, status) %>% 
  group_by(status) %>% 
  count()
)

map(test_scaled, ~select(.x, status) %>% 
      group_by(status) %>% 
      count()
)

train_scaled[[1]]
#################################################################
# One input layer with 17 neurons (one for each feature)
# One hidden layer with 17 neurons
# One output layer with 2 neurons (one for each class)
# We will have 156 parameters/nodes in the model: (17*17) + 17 + (17*2) + 2

library(kerasR)

NN_units <- ncol(X_train_scaled[[1]]) # Number of input variables
NN_model <- keras_model_sequential() %>% 
  layer_dense(units = NN_units, activation = 'relu', input_shape = NN_units) %>% 
  layer_dense(units = 2, activation = 'sigmoid')

NN_model %>% 
  summary

NN_model %>% 
  compile(
    loss = 'binary_crossentropy',
    #optimizer_adam(lr = 0.01),
    optimizer_sgd(lr = 0.01, momentum = 0.9),
    metrics = c('accuracy')
  )

history <- future_map2(
  .x = X_train_scaled,
  .y = Y_train_scaled_cat,
  ~fit(
    NN_model,
    x = .x,
    y = .y,
    epochs = 50,
    batch_size = 20,
    validation_split = 0
  )
)

#cowplot::plot_grid(plot(history[[1]]), plot(history[[2]]), plot(history[[3]]), plot(history[[4]]))

history_data <- map_dfr(history, as.data.frame, .id = 'Model')

map2(
  .x = X_test_scaled,
  .y = Y_test_scaled_cat,
  ~evaluate(NN_model, .x, .y)
)

results_NN <- map2(
  .x = test_scaled,
  .y = X_test_scaled,
  ~mutate(.x,
          pred_status = predict_classes(NN_model, .y),
          correct = case_when(
            status == pred_status ~ "Correct",
            TRUE ~ "Incorrect"
          ),
          status_text = case_when(
            status == 1 ~ "Bankrupt",
            status == 0 ~ "Non-Bankrupt"
          )
  ) %>% 
    mutate_at(vars(status, pred_status, correct, status_text), funs(factor)) %>% 
    add_count(status, pred_status)
)

conMat_NN <- results_NN %>%
  future_map(.,
             ~confusionMatrix(
               as.factor(..1$pred_status),
               as.factor(..1$status),
               mode = "everything",
               positive = "1"
             )
  )
################ Create Table for ConMat ##########################

conMatTable <- list(
  TP <- conMat_NN %>% 
    map(., 
        ~as.numeric(.x$table[[1]])
    ),
  FP <- conMat_NN %>% 
    map(.,
        ~as.numeric(.x$table[[2]])
    ),
  FN <- conMat_NN %>% 
    map(.,
        ~as.numeric(.x$table[[3]])
    ),
  TN <- conMat_NN %>% 
    map(.,
        ~as.numeric(.x$table[[4]])
    )
)

# MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
MCC_NN <- pmap_dbl(conMatTable, ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))
MCC_NN

AUCs_NN <- results_NN %>%
  map(., ~pROC::auc(as.numeric(.x$status), as.numeric(.x$pred_status)))

#####################################################################

fg_NN <- results_NN %>% 
  map(., ~filter(.x, status == 1) %>% 
        pull(pred_status))

bg_NN <- results_NN %>% 
  map(., ~filter(.x, status == 0) %>% 
        pull(pred_status))

pr_NN <- map2(
  .x = fg_NN,
  .y = bg_NN,
  ~pr.curve(
    scores.class0 = .x,
    scores.class1 = .y,
    curve = TRUE)
)

AUPRCs_NN <- pr_NN %>% 
  map(., pluck, 'auc.integral')

Sums_NN <- map(conMat_NN, pluck, 'table') %>% 
  map(., ~sum(.[1:4]))

#####################################################################
conMat_NN[[2]]$table
AUPRCs_NN[[1]][1]
sum(conMat_NN[[4]]$table[1:4])

LaTeX_Table_Function <- function(L){
  table = data.frame(
    Accuracy = conMat_NN[[L]]$overall[[1]],
    Sensitivity = conMat_NN[[L]]$byClass[[1]],
    Specificity = conMat_NN[[L]]$byClass[[2]],
    Precision = conMat_NN[[L]]$byClass[[5]],
    F1 = conMat_NN[[L]]$byClass[[7]],
    MCC = MCC_NN[[L]],
    AUC = AUCs_NN[[L]][[1]],
    AUPRC = AUPRCs_NN[[L]],
    Total = Sums_NN[[L]]
  ) %>% 
    t()
  return(table)
}

LaTeX_Table_Results <- lapply(seq(1:4), LaTeX_Table_Function)
LaTeX_Table_Results %>% 
  map(., ~rownames_to_column(as.data.frame(.x), "Metric")) %>% 
  reduce(inner_join, by = 'Metric') %>% 
  setNames(c("Metric", "1 Year", "2 Year", "3 Year", "4 Year")) %>% 
  stargazer(summary = FALSE, digits = 2, rownames = FALSE, colnames = FALSE)


#################################################################
################# Deep NN Model #################################
#################################################################

# Deep learning NN model

NN_units <- ncol(X_train_scaled[[1]]) # Number of input variables
NN_model_deep1 <- keras_model_sequential() %>% 
  layer_dense(units = NN_units*2, activation = 'relu', input_shape = NN_units) %>% 
  layer_dropout(rate = 0.3) %>% # prevents overfitting
  layer_dense(units = 20, activation = 'relu', input_shape = NN_units) %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 2, activation = 'sigmoid')

NN_model_deep1 %>% 
  summary()

NN_model_deep1 %>% 
  compile(
    loss = 'binary_crossentropy',
    #optimizer_adam(lr = 0.01),
    optimizer_sgd(lr = 0.01, momentum = 0.9),
    metrics = c('accuracy')
  )

history_deep1 <- future_map2(
  .x = X_train_scaled,
  .y = Y_train_scaled_cat,
  ~fit(
    NN_model_deep1,
    x = .x,
    y = .y,
    epochs = 50,
    batch_size = 20,
    validation_split = 0
  )
)

history_data_deep1 <- map_dfr(history_deep1, as.data.frame, .id = 'Model')


map2(
  .x = X_test_scaled,
  .y = Y_test_scaled_cat,
  ~evaluate(NN_model_deep1, .x, .y)
)

results_NN_deep1 <- map2(
  .x = test_scaled,
  .y = X_test_scaled,
  ~mutate(.x,
          pred_status = predict_classes(NN_model_deep1, .y),
          correct = case_when(
            status == pred_status ~ "Correct",
            TRUE ~ "Incorrect"
          ),
          status_text = case_when(
            status == 1 ~ "Bankrupt",
            status == 0 ~ "Non-Bankrupt"
          )
  ) %>% 
    mutate_at(vars(status, pred_status, correct, status_text), funs(factor)) %>% 
    add_count(status, pred_status)
)


conMat_NN_deep1 <- results_NN_deep1 %>%
  future_map(.,
             ~confusionMatrix(
               as.factor(..1$pred_status),
               as.factor(..1$status),
               mode = "everything",
               positive = "1"
             )
  )
################ Create Table for ConMat ##########################

conMatTable <- list(
  TP <- conMat_NN_deep1 %>% 
    map(., 
        ~as.numeric(.x$table[[1]])
    ),
  FP <- conMat_NN_deep1 %>% 
    map(.,
        ~as.numeric(.x$table[[2]])
    ),
  FN <- conMat_NN_deep1 %>% 
    map(.,
        ~as.numeric(.x$table[[3]])
    ),
  TN <- conMat_NN_deep1 %>% 
    map(.,
        ~as.numeric(.x$table[[4]])
    )
)

# MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
MCC_NN_deep1 <- pmap_dbl(conMatTable, ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))
MCC_NN_deep1

AUCs_NN_deep1 <- results_NN_deep1 %>%
  map(., ~pROC::auc(as.numeric(.x$status), as.numeric(.x$pred_status)))

#####################################################################

fg_NN_deep1 <- results_NN_deep1 %>% 
  map(., ~filter(.x, status == 1) %>% 
        pull(pred_status))

bg_NN_deep1 <- results_NN_deep1 %>% 
  map(., ~filter(.x, status == 0) %>% 
        pull(pred_status))

pr_NN_deep1 <- map2(
  .x = fg_NN_deep1,
  .y = bg_NN_deep1,
  ~pr.curve(
    scores.class0 = .x,
    scores.class1 = .y,
    curve = TRUE)
)

AUPRCs_NN_deep1 <- pr_NN_deep1 %>% 
  map(., pluck, 'auc.integral')

Sums_NN_deep1 <- map(conMat_NN_deep1, pluck, 'table') %>% 
  map(., ~sum(.[1:4]))

#####################################################################
conMat_NN_deep1[[2]]$table
AUPRCs_NN_deep1[[1]][1]
sum(conMat_NN_deep1[[4]]$table[1:4])

LaTeX_Table_Function <- function(L){
  table = data.frame(
    Accuracy = conMat_NN_deep1[[L]]$overall[[1]],
    Sensitivity = conMat_NN_deep1[[L]]$byClass[[1]],
    Specificity = conMat_NN_deep1[[L]]$byClass[[2]],
    Precision = conMat_NN_deep1[[L]]$byClass[[5]],
    F1 = conMat_NN_deep1[[L]]$byClass[[7]],
    MCC = MCC_NN_deep1[[L]],
    AUC = AUCs_NN_deep1[[L]][[1]],
    AUPRC = AUPRCs_NN_deep1[[L]],
    Total = Sums_NN_deep1[[L]]
  ) %>% 
    t()
  return(table)
}

LaTeX_Table_Results <- lapply(seq(1:4), LaTeX_Table_Function)
LaTeX_Table_Results %>% 
  map(., ~rownames_to_column(as.data.frame(.x), "Metric")) %>% 
  reduce(inner_join, by = 'Metric') %>% 
  setNames(c("Metric", "1 Year", "2 Year", "3 Year", "4 Year")) %>% 
  stargazer(summary = FALSE, digits = 2, rownames = FALSE, colnames = FALSE)


#########################################################


#################################################################
################# Compare with XGBoost ##########################
#################################################################
params <- list(
  "objective"="binary:logistic",
  #"eval_metric"= "auc",
  "eta" = 0.1,
  "gamma" = 0.5,
  "lambda" = 1,
  "alpha" = 0,
  "min_child_weight" = 5,
  "max_delta_step" = 0,
  "max_depth" = 5,
  "colsample_bytree" = 0.75,
  "subsample"= 1,
  "colsample_bylevel" = 1,
  "scale_pos_weight" = weights,
  "set.seed" = 176
)

NRounds = 88

Y_train_scaled <- train_scaled %>%
  map(., ~pull(., status))

dtrain_scaled <- future_map2(
  X_train_scaled,
  Y_train_scaled, 
  ~xgb.DMatrix(data = .x, label = .y))

dtest_scaled <- future_map(
  X_test_scaled,
  ~xgb.DMatrix(data = .x))

#################################################################

weights = train_scaled %>% 
  future_map(.,
             ~group_by(., status) %>% 
               summarise(status_sums = n()) %>% 
               pivot_wider(names_from = status, values_from = status_sums) %>%  
               mutate(active_over_bankrupt = `0` / `1`) %>% 
               pull(active_over_bankrupt)
  )

xgb.model_NArm_auc <- dtrain_scaled %>% 
  future_map(.,
             ~xgboost(
               params = params,
               "eval_metric" = "auc",
               data = .x,
               nrounds = NRounds,
               nthread = parallel::detectCores()
             )
  )

xgb.model_NArm_logloss <- dtrain_scaled %>% 
  future_map(.,
             ~xgboost(
               params = params,
               "eval_metric" = "logloss",
               data = .x,
               nrounds = NRounds,
               nthread = parallel::detectCores()
             )
  )

XGB_eval_log <- bind_cols(
  map_dfr(xgb.model_NArm_auc, pluck, 'evaluation_log', .id = 'Model') %>% 
    setNames(c("iter_auc", "train_auc", "Model_auc")),
  map_dfr(xgb.model_NArm_logloss, pluck, 'evaluation_log', .id = 'Model') %>% 
    setNames(c("iter_logloss", "train_logloss", "Model_logloss"))
)



#################################################################

predictions_XGB_NArm <- future_map2(
  xgb.model_NArm_auc,
  dtest_scaled,
  ~predict(
    object = .x,
    newdata = .y,
    type = 'prob') 
)

results_XGB <- future_map2(
  test_scaled,
  predictions_XGB_NArm,
  ~cbind(.x, .y)
) %>% 
  future_map(.,
             ~mutate(.,
                     pred_status = case_when(
                       .y > 0.50 ~ 1,
                       .y <= 0.50 ~ 0
                     ),
                     correct = case_when(
                       status == pred_status ~ "Correct",
                       TRUE ~ "Incorrect"
                     ),
                     status_text = case_when(
                       status == 1 ~ "Bankrupt",
                       status == 0 ~ "Non-Bankrupt"
                     )
             ) %>% 
               rename(pred = .y) %>% 
               mutate_at(vars(status, pred_status, correct, status_text), funs(factor)) %>% 
               add_count(status, pred_status)
  ) 


conMat_XGB <- results_XGB %>%
  future_map(.,
             ~confusionMatrix(
               as.factor(..1$pred_status),
               as.factor(..1$status),
               mode = "everything",
               positive = "1"
             )
  )

################ Create Table for ConMat ##########################

conMatTable <- list(
  TP <- conMat_XGB %>% 
    map(., 
        ~as.numeric(.x$table[[1]])
    ),
  FP <- conMat_XGB %>% 
    map(.,
        ~as.numeric(.x$table[[2]])
    ),
  FN <- conMat_XGB %>% 
    map(.,
        ~as.numeric(.x$table[[3]])
    ),
  TN <- conMat_XGB %>% 
    map(.,
        ~as.numeric(.x$table[[4]])
    )
)

# MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
MCC_XGB <- pmap_dbl(conMatTable, ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))
MCC_XGB

AUCs_XGB <- results_XGB %>%
  map(., ~pROC::auc(as.numeric(.x$status), as.numeric(.x$pred_status)))

#####################################################################

fg_XGB <- results_XGB %>% 
  map(., ~filter(.x, status == 1) %>% 
        pull(pred_status))

bg_XGB <- results_XGB %>% 
  map(., ~filter(.x, status == 0) %>% 
        pull(pred_status))

pr_XGB <- map2(
  .x = fg_XGB,
  .y = bg_XGB,
  ~pr.curve(
    scores.class0 = .x,
    scores.class1 = .y,
    curve = TRUE)
)

AUPRCs_XGB <- pr_XGB %>% 
  map(., pluck, 'auc.integral')

Sums_XGB <- map(conMat_XGB, pluck, 'table') %>% 
  map(., ~sum(.[1:4]))
#####################################################################

LaTeX_Table_Function <- function(L){
  table = data.frame(
    Accuracy = conMat_XGB[[L]]$overall[[1]],
    Sensitivity = conMat_XGB[[L]]$byClass[[1]],
    Specificity = conMat_XGB[[L]]$byClass[[2]],
    Precision = conMat_XGB[[L]]$byClass[[5]],
    F1 = conMat_XGB[[L]]$byClass[[7]],
    MCC = MCC_XGB[[L]],
    AUC = AUCs_XGB[[L]][[1]],
    AUPRC = AUPRCs_XGB[[L]],
    Total = Sums_XGB[[L]]
  ) %>% 
    t()
  return(table)
}

LaTeX_Table_Results <- lapply(seq(1:4), LaTeX_Table_Function)
LaTeX_Table_Results %>% 
  map(., ~rownames_to_column(as.data.frame(.x), "Metric")) %>% 
  reduce(inner_join, by = 'Metric') %>% 
  setNames(c("Metric", "1 Year", "2 Year", "3 Year", "4 Year")) %>% 
  stargazer(summary = FALSE, digits = 2, rownames = FALSE, colnames = FALSE)

#################################################################
################# Logistic Regression ###########################
#################################################################

XY_vars <- train_scaled[[1]] %>% 
  select(-ID, -Postcode, -City, -Latitude, -Longitude, -Region.in.country,
         -Major.sectors, -NACE.Rev..2.main.section, -BvD.ID.number)

X_formula <- str_subset(names(XY_vars), "status", negate = TRUE)
FORMULA <- reformulate(X_formula, response = "status")

Logit.model_NArm <- train_scaled %>% 
  future_map(., ~glm(formula = FORMULA, data = ., family = "binomial"))

Logit.model_NArm %>% 
  map(., ~summary(.))

#################################################################
stargazer(Logit.model_NArm, title = "Logistic Regression Results",
          align = TRUE, no.space = TRUE, font.size = "footnotesize",
          dep.var.labels = c("Binary Status of Bankruptcy"),
          column.labels = c(" 1 Year", "2 Year", "3 Year", "4 Year")
)
#################################################################

predictions_Logit_NArm <- future_map2(
  Logit.model_NArm,
  test_scaled,
  ~predict(
    object = .x,
    newdata = .y,
    type = 'response'
  )
)

results_Logistic <- future_map2(
  test_scaled,
  predictions_Logit_NArm,
  ~cbind(.x, .y)
) %>% 
  future_map(.,
             ~mutate(.,
                     pred_status = case_when(
                       .y > 0.50 ~ 1,
                       .y <= 0.50 ~ 0
                     ),
                     correct = case_when(
                       status == pred_status ~ "Correct",
                       TRUE ~ "Incorrect"
                     ),
                     status_text = case_when(
                       status == 1 ~ "Bankrupt",
                       status == 0 ~ "Non-Bankrupt"
                     )
             ) %>% 
               rename(pred = .y) %>% 
               mutate_at(vars(status, pred_status, correct, status_text), funs(factor)) %>% 
               add_count(status, pred_status)
  ) 

conMat_Logistic <- results_Logistic %>%
  future_map(.,
             ~confusionMatrix(
               as.factor(..1$pred_status),
               as.factor(..1$status),
               mode = "everything",
               positive = "1"
             )
  )

################ Create Table for ConMat ##########################

conMatTable <- list(
  TP <- conMat_Logistic %>% 
    map(., 
        ~as.numeric(.x$table[[1]])
    ),
  FP <- conMat_Logistic %>% 
    map(.,
        ~as.numeric(.x$table[[2]])
    ),
  FN <- conMat_Logistic %>% 
    map(.,
        ~as.numeric(.x$table[[3]])
    ),
  TN <- conMat_Logistic %>% 
    map(.,
        ~as.numeric(.x$table[[4]])
    )
)

# MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
MCC_Logit <- pmap_dbl(conMatTable, ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))
MCC_Logit

AUCs_Logit <- results_XGB %>%
  map(., ~pROC::auc(as.numeric(.x$status), as.numeric(.x$pred_status)))

#####################################################################

fg_Logit <- results_Logistic %>% 
  map(., ~filter(.x, status == 1) %>% 
        pull(pred_status))

bg_Logit <- results_Logistic %>% 
  map(., ~filter(.x, status == 0) %>% 
        pull(pred_status))

pr_Logit <- map2(
  .x = fg_Logit,
  .y = bg_Logit,
  ~pr.curve(
    scores.class0 = .x,
    scores.class1 = .y,
    curve = TRUE)
)

AUPRCs_Logit <- pr_Logit %>% 
  map(., pluck, 'auc.integral')

Sums_Logit <- map(conMat_Logistic, pluck, 'table') %>% 
  map(., ~sum(.[1:4]))
#####################################################################

LaTeX_Table_Function <- function(L){
  table = data.frame(
    Accuracy = conMat_Logistic[[L]]$overall[[1]],
    Sensitivity = conMat_Logistic[[L]]$byClass[[1]],
    Specificity = conMat_Logistic[[L]]$byClass[[2]],
    Precision = conMat_Logistic[[L]]$byClass[[5]],
    F1 = conMat_Logistic[[L]]$byClass[[7]],
    MCC = MCC_Logit[[L]],
    AUC = AUCs_Logit[[L]][[1]],
    AUPRC = AUPRCs_Logit[[L]],
    Total = Sums_Logit[[L]]
  ) %>% 
    t()
  return(table)
}

LaTeX_Table_Results <- lapply(seq(1:4), LaTeX_Table_Function)
LaTeX_Table_Results %>% 
  map(., ~rownames_to_column(as.data.frame(.x), "Metric")) %>% 
  reduce(inner_join, by = 'Metric') %>% 
  setNames(c("Metric", "1 Year", "2 Year", "3 Year", "4 Year")) %>% 
  stargazer(summary = FALSE, digits = 2, rownames = FALSE, colnames = FALSE)

#################################################################
################# SVM Model (Radial Kernel) #####################
#################################################################
cat(X_formula, sep = " + ")  # NOTE: If we add variables we need to paste this into below

############## SVM Optimal parameter s###########################
svm_opt_params <- train_scaled %>% 
  map(.,
      ~tune(
        svm,
        train.x = factor(status) ~ TL.TA + CA.CL + TL.EQ + WC.EBIT + SALES.EBIT + 
          CL.FinExp + EQ.Turnover + CF.NCL + logTA + logSALES + CF.CL + 
          CF.SALES + DEBTORS.SALES,
        data = .,
        ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)),
        tunecontrol = tune.control(sampling = "fix")
      )
  )

##################### SVM Radial ###################################
svm.model_NArm <- future_map2(
  .x = train_scaled,
  .y = svm_opt_params,
  ~svm(
    formula = factor(status) ~ TL.TA + CA.CL + TL.EQ + WC.EBIT + SALES.EBIT + 
      CL.FinExp + EQ.Turnover + CF.NCL + logTA + logSALES + CF.CL + 
      CF.SALES + DEBTORS.SALES,
    data = .x,
    type = "C-classification",
    kernel = "radial", degree = 2, probability = TRUE,
    gamma = .y$best.parameters$gamma, cost = .y$best.parameters$cost
  )
)

predictions_SVM_NArm <- future_map2(
  svm.model_NArm,
  test_scaled,
  ~predict(
    object = .x,
    newdata = .y,
    probability = TRUE
  )
)

results_SVM <- future_map2(
  test_scaled,
  map(predictions_SVM_NArm, ~attributes(.)$probabilities),
  ~cbind(.x, .y)
) %>% 
  future_map(.,
             ~mutate(.,
                     pred_prob = `1`,
                     pred_status = case_when(
                       pred_prob > 0.50 ~ 1,
                       pred_prob <= 0.50 ~ 0
                     ),
                     correct = case_when(
                       status == pred_status ~ "Correct",
                       TRUE ~ "Incorrect"
                     ),
                     status_text = case_when(
                       status == 1 ~ "Bankrupt",
                       status == 0 ~ "Non-Bankrupt"
                     )
             ) %>% 
               rename(pred = pred_prob) %>% 
               mutate_at(vars(status, pred_status, correct, status_text), funs(factor)) %>% 
               add_count(status, pred_status)
  ) 



conMat_SVM <- results_SVM %>%
  future_map(.,
             ~confusionMatrix(
               as.factor(..1$pred_status),
               as.factor(..1$status),
               mode = "everything",
               positive = "1"
             )
  )

################ Create Table for ConMat ##########################

conMatTable <- list(
  TP <- conMat_SVM %>% 
    map(., 
        ~as.numeric(.x$table[[1]])
    ),
  FP <- conMat_SVM %>% 
    map(.,
        ~as.numeric(.x$table[[2]])
    ),
  FN <- conMat_SVM %>% 
    map(.,
        ~as.numeric(.x$table[[3]])
    ),
  TN <- conMat_SVM %>% 
    map(.,
        ~as.numeric(.x$table[[4]])
    )
)

# MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
MCC_SVM <- pmap_dbl(conMatTable, ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))
MCC_SVM

AUCs_SVM <- results_SVM %>%
  map(., ~pROC::auc(as.numeric(.x$status), as.numeric(.x$pred_status)))

#####################################################################

fg_SVM <- results_SVM %>% 
  map(., ~filter(.x, status == 1) %>% 
        pull(pred_status))

bg_SVM <- results_SVM %>% 
  map(., ~filter(.x, status == 0) %>% 
        pull(pred_status))

pr_SVM <- map2(
  .x = fg_SVM,
  .y = bg_SVM,
  ~pr.curve(
    scores.class0 = .x,
    scores.class1 = .y,
    curve = TRUE)
)

AUPRCs_SVM <- pr_SVM %>% 
  map(., pluck, 'auc.integral')

Sums_SVM <- map(conMat_SVM, pluck, 'table') %>% 
  map(., ~sum(.[1:4]))
#####################################################################

LaTeX_Table_Function <- function(L){
  table = data.frame(
    Accuracy = conMat_SVM[[L]]$overall[[1]],
    Sensitivity = conMat_SVM[[L]]$byClass[[1]],
    Specificity = conMat_SVM[[L]]$byClass[[2]],
    Precision = conMat_SVM[[L]]$byClass[[5]],
    F1 = conMat_SVM[[L]]$byClass[[7]],
    MCC = MCC_SVM[[L]],
    AUC = AUCs_SVM[[L]][[1]],
    AUPRC = AUPRCs_SVM[[L]],
    Total = Sums_SVM[[L]]
  ) %>% 
    t()
  return(table)
}

LaTeX_Table_Results <- lapply(seq(1:4), LaTeX_Table_Function)
LaTeX_Table_Results %>% 
  map(., ~rownames_to_column(as.data.frame(.x), "Metric")) %>% 
  reduce(inner_join, by = 'Metric') %>% 
  setNames(c("Metric", "1 Year", "2 Year", "3 Year", "4 Year")) %>% 
  stargazer(summary = FALSE, digits = 2, rownames = FALSE, colnames = FALSE)

#################################################################
################# SVM Model Linear Kernel #######################
#################################################################
cat(X_formula, sep = " + ")  # NOTE: If we add variables we need to paste this into below

svm.model_NArm_linear <- future_map2(
  .x = train_scaled,
  .y = svm_opt_params,
  ~svm(
    formula = factor(status) ~ TL.TA + CA.CL + TL.EQ + WC.EBIT + SALES.EBIT 
    + CL.FinExp + EQ.Turnover + CF.NCL + logTA + logSALES + CF.CL +
      CF.SALES + DEBTORS.SALES,
    data = .x,
    type = "C-classification",
    kernel = "linear", degree = 2, probability = TRUE,
    gamma = .y$best.parameters$gamma, cost = .y$best.parameters$cost
  )
)

predictions_SVM_NArm_linear <- future_map2(
  svm.model_NArm_linear,
  test_scaled,
  ~predict(
    object = .x,
    newdata = .y,
    probability = TRUE
  )
)

results_SVM_linear <- future_map2(
  test_scaled,
  map(predictions_SVM_NArm_linear, ~attributes(.)$probabilities),
  ~cbind(.x, .y)
) %>% 
  future_map(.,
             ~mutate(.,
                     pred_prob = `1`,
                     pred_status = case_when(
                       pred_prob > 0.50 ~ 1,
                       pred_prob <= 0.50 ~ 0
                     ),
                     correct = case_when(
                       status == pred_status ~ "Correct",
                       TRUE ~ "Incorrect"
                     ),
                     status_text = case_when(
                       status == 1 ~ "Bankrupt",
                       status == 0 ~ "Non-Bankrupt"
                     )
             ) %>% 
               rename(pred = pred_prob) %>% 
               mutate_at(vars(status, pred_status, correct, status_text), funs(factor)) %>% 
               add_count(status, pred_status)
  ) 

results_SVM_linear <- results_SVM_linear[1:3]   # NOTE: The 4 years prior predictions 
# fail to make any bankrupt predictions and returns NA lists

conMat_SVM_linear <- results_SVM_linear %>%
  future_map(.,
             ~confusionMatrix(
               as.factor(..1$pred_status),
               as.factor(..1$status),
               mode = "everything",
               positive = "1"
             )
  )

################ Create Table for ConMat ##########################

conMatTable <- list(
  TP <- conMat_SVM_linear %>% 
    map(., 
        ~as.numeric(.x$table[[1]])
    ),
  FP <- conMat_SVM_linear %>% 
    map(.,
        ~as.numeric(.x$table[[2]])
    ),
  FN <- conMat_SVM_linear %>% 
    map(.,
        ~as.numeric(.x$table[[3]])
    ),
  TN <- conMat_SVM_linear %>% 
    map(.,
        ~as.numeric(.x$table[[4]])
    )
)

# MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
MCC_SVM_linear <- pmap_dbl(conMatTable, ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))
MCC_SVM_linear

AUCs_SVM_linear <- results_SVM_linear %>%
  map(., ~pROC::auc(as.numeric(.x$status), as.numeric(.x$pred_status)))

#####################################################################

fg_SVM_linear <- results_SVM_linear %>% 
  map(., ~filter(.x, status == 1) %>% 
        pull(pred_status))

bg_SVM_linear <- results_SVM_linear %>% 
  map(., ~filter(.x, status == 0) %>% 
        pull(pred_status))

pr_SVM_linear <- map2(
  .x = fg_SVM_linear,
  .y = bg_SVM_linear,
  ~pr.curve(
    scores.class0 = .x,
    scores.class1 = .y,
    curve = TRUE)
)

AUPRCs_SVM_linear <- pr_SVM_linear %>% 
  map(., pluck, 'auc.integral')

Sums_SVM_linear <- map(conMat_SVM_linear, pluck, 'table') %>% 
  map(., ~sum(.[1:4]))
#####################################################################

LaTeX_Table_Function <- function(L){
  table = data.frame(
    Accuracy = conMat_SVM_linear[[L]]$overall[[1]],
    Sensitivity = conMat_SVM_linear[[L]]$byClass[[1]],
    Specificity = conMat_SVM_linear[[L]]$byClass[[2]],
    Precision = conMat_SVM_linear[[L]]$byClass[[5]],
    F1 = conMat_SVM_linear[[L]]$byClass[[7]],
    MCC = MCC_SVM_linear[[L]],
    AUC = AUCs_SVM_linear[[L]][[1]],
    AUPRC = AUPRCs_SVM_linear[[L]],
    Total = Sums_SVM_linear[[L]]
  ) %>% 
    t()
  return(table)
}

LaTeX_Table_Results <- lapply(seq(1:3), LaTeX_Table_Function)
LaTeX_Table_Results %>% 
  map(., ~rownames_to_column(as.data.frame(.x), "Metric")) %>% 
  reduce(inner_join, by = 'Metric') %>% 
  setNames(c("Metric", "1 Year", "2 Year", "3 Year")) %>% 
  stargazer(summary = FALSE, digits = 2, rownames = FALSE, colnames = FALSE)

#################################################################
#################################################################
#################################################################
################# Random Forest #################################
#################################################################
cat(X_formula, sep = " + ")  # NOTE: If we add variables we need to paste this into below

randomForest.model_NArm <- train_scaled %>% 
  future_map(., ~randomForest::randomForest(factor(status) ~ TL.TA + CA.CL + TL.EQ + WC.EBIT + SALES.EBIT + 
                                              CL.FinExp + EQ.Turnover + CF.NCL + logTA + logSALES + CF.CL +
                                              CF.SALES + DEBTORS.SALES,
                                            data = .))


predictions_randomForest_NArm <- future_map2(
  randomForest.model_NArm,
  test_scaled,
  ~predict(
    object = .x,
    newdata = .y,
    type = 'prob'
  )
)

results_randomForest <- future_map2(
  test_scaled,
  predictions_randomForest_NArm,
  ~cbind(.x, .y)
) %>% 
  future_map(.,
             ~mutate(.,
                     pred_prob = `1`,
                     pred_status = case_when(
                       pred_prob > 0.50 ~ 1,
                       pred_prob <= 0.50 ~ 0
                     ),
                     correct = case_when(
                       status == pred_status ~ "Correct",
                       TRUE ~ "Incorrect"
                     ),
                     status_text = case_when(
                       status == 1 ~ "Bankrupt",
                       status == 0 ~ "Non-Bankrupt"
                     )
             ) %>% 
               rename(pred = pred_prob) %>% 
               mutate_at(vars(status, pred_status, correct, status_text), funs(factor)) %>% 
               add_count(status, pred_status)
  ) 


conMat_randomForest <- results_randomForest %>%
  future_map(.,
             ~confusionMatrix(
               as.factor(..1$pred_status),
               as.factor(..1$status),
               mode = "everything",
               positive = "1"
             )
  )

################ Create Table for ConMat ##########################

conMatTable <- list(
  TP <- conMat_randomForest %>% 
    map(., 
        ~as.numeric(.x$table[[1]])
    ),
  FP <- conMat_randomForest %>% 
    map(.,
        ~as.numeric(.x$table[[2]])
    ),
  FN <- conMat_randomForest %>% 
    map(.,
        ~as.numeric(.x$table[[3]])
    ),
  TN <- conMat_randomForest %>% 
    map(.,
        ~as.numeric(.x$table[[4]])
    )
)

# MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
MCC_RF <- pmap_dbl(conMatTable, ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))
MCC_RF

AUCs_RF <- results_randomForest %>%
  map(., ~pROC::auc(as.numeric(.x$status), as.numeric(.x$pred_status)))

#####################################################################

fg_RF <- results_randomForest %>% 
  map(., ~filter(.x, status == 1) %>% 
        pull(pred_status))

bg_RF <- results_randomForest %>% 
  map(., ~filter(.x, status == 0) %>% 
        pull(pred_status))

pr_RF <- map2(
  .x = fg_RF,
  .y = bg_RF,
  ~pr.curve(
    scores.class0 = .x,
    scores.class1 = .y,
    curve = TRUE)
)

AUPRCs_RF <- pr_RF %>% 
  map(., pluck, 'auc.integral')

Sums_RF <- map(conMat_randomForest, pluck, 'table') %>% 
  map(., ~sum(.[1:4]))
#####################################################################

LaTeX_Table_Function <- function(L){
  table = data.frame(
    Accuracy = conMat_randomForest[[L]]$overall[[1]],
    Sensitivity = conMat_randomForest[[L]]$byClass[[1]],
    Specificity = conMat_randomForest[[L]]$byClass[[2]],
    Precision = conMat_randomForest[[L]]$byClass[[5]],
    F1 = conMat_randomForest[[L]]$byClass[[7]],
    MCC = MCC_RF[[L]],
    AUC = AUCs_RF[[L]][[1]],
    AUPRC = AUPRCs_RF[[L]],
    Total = Sums_RF[[L]]
  ) %>% 
    t()
  return(table)
}

LaTeX_Table_Results <- lapply(seq(1:4), LaTeX_Table_Function)
LaTeX_Table_Results %>% 
  map(., ~rownames_to_column(as.data.frame(.x), "Metric")) %>% 
  reduce(inner_join, by = 'Metric') %>% 
  setNames(c("Metric", "1 Year", "2 Year", "3 Year", "4 Year")) %>% 
  stargazer(summary = FALSE, digits = 2, rownames = FALSE, colnames = FALSE)

#################################################################
################# LightGBM Model ################################
#################################################################
lgbm_train <- map2(
  .x = X_train_scaled,
  .y = Y_train_scaled,
  ~lgb.Dataset(data = .x, label = .y)
)

# NOTE: Add scale pos weight for lightGBM into parameter list

# lgbm_test <- map(
#   .x = X_test_scaled,
#   ~lgb.Dataset(data = .x)
# )

params_lightGBM <- list(
  objective = "binary",
  metric = "auc",
  min_sum_hessian_in_leaf = 1,
  feature_fraction = 0.7,
  bagging_fraction = 0.7,
  bagging_freq = 5,
  min_data = 100,
  max_bin = 50,
  lambda_l1 = 8,
  lambda_l2 = 1.3,
  min_data_in_bin=100,
  min_gain_to_split = 10,
  min_data_in_leaf = 30,
  is_unbalance = TRUE
)

lightGBM.model_NArm <- lgbm_train %>% 
  map(., 
      ~lgb.train(
        data = .,
        params = params_lightGBM,
        learning_rate = 0.1
      )
  )

predictions_lightGBM_NArm <- future_map2(
  lightGBM.model_NArm,
  X_test_scaled,
  ~predict(
    object = .x,
    data = .y,
    rawscore = FALSE) # setting rawscore=TRUE for logistic regression would result in predictions for log-odds instead of probabilities
)

results_lightGBM <- future_map2(
  test_scaled,
  predictions_lightGBM_NArm,
  ~cbind(.x, .y)
) %>%  
  future_map(.,
             ~mutate(.,
                     pred_status = case_when(
                       .y > 0.50 ~ 1,
                       .y <= 0.50 ~ 0
                     ),
                     correct = case_when(
                       status == pred_status ~ "Correct",
                       TRUE ~ "Incorrect"
                     ),
                     status_text = case_when(
                       status == 1 ~ "Bankrupt",
                       status == 0 ~ "Non-Bankrupt"
                     )
             ) %>% 
               rename(pred = .y) %>% 
               mutate_at(vars(status, pred_status, correct, status_text), funs(factor)) %>% 
               add_count(status, pred_status)
  ) 


conMat_lightGBM <- results_lightGBM %>%
  future_map(.,
             ~confusionMatrix(
               as.factor(..1$pred_status),
               as.factor(..1$status),
               mode = "everything",
               positive = "1"
             )
  )


################ Create Table for ConMat ##########################

conMatTable <- list(
  TP <- conMat_lightGBM %>% 
    map(., 
        ~as.numeric(.x$table[[1]])
    ),
  FP <- conMat_lightGBM %>% 
    map(.,
        ~as.numeric(.x$table[[2]])
    ),
  FN <- conMat_lightGBM %>% 
    map(.,
        ~as.numeric(.x$table[[3]])
    ),
  TN <- conMat_lightGBM %>% 
    map(.,
        ~as.numeric(.x$table[[4]])
    )
)

# MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
MCC_lightGBM <- pmap_dbl(conMatTable, ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))
MCC_lightGBM

AUCs_lightGBM <- results_lightGBM %>%
  map(., ~pROC::auc(as.numeric(.x$status), as.numeric(.x$pred_status)))

#####################################################################

fg_lightGBM <- results_lightGBM %>% 
  map(., ~filter(.x, status == 1) %>% 
        pull(pred_status))

bg_lightGBM <- results_lightGBM %>% 
  map(., ~filter(.x, status == 0) %>% 
        pull(pred_status))

pr_lightGBM <- map2(
  .x = fg_lightGBM,
  .y = bg_lightGBM,
  ~pr.curve(
    scores.class0 = .x,
    scores.class1 = .y,
    curve = TRUE)
)

AUPRCs_lightGBM <- pr_lightGBM %>% 
  map(., pluck, 'auc.integral')

Sums_lightGBM <- map(conMat_lightGBM, pluck, 'table') %>% 
  map(., ~sum(.[1:4]))
#####################################################################

LaTeX_Table_Function <- function(L){
  table = data.frame(
    Accuracy = conMat_lightGBM[[L]]$overall[[1]],
    Sensitivity = conMat_lightGBM[[L]]$byClass[[1]],
    Specificity = conMat_lightGBM[[L]]$byClass[[2]],
    Precision = conMat_lightGBM[[L]]$byClass[[5]],
    F1 = conMat_lightGBM[[L]]$byClass[[7]],
    MCC = MCC_lightGBM[[L]],
    AUC = AUCs_lightGBM[[L]][[1]],
    AUPRC = AUPRCs_lightGBM[[L]],
    Total = Sums_lightGBM[[L]]
  ) %>% 
    t()
  return(table)
}

LaTeX_Table_Results <- lapply(seq(1:4), LaTeX_Table_Function)
LaTeX_Table_Results %>% 
  map(., ~rownames_to_column(as.data.frame(.x), "Metric")) %>% 
  reduce(inner_join, by = 'Metric') %>% 
  setNames(c("Metric", "1 Year", "2 Year", "3 Year", "4 Year")) %>% 
  stargazer(summary = FALSE, digits = 2, rownames = FALSE, colnames = FALSE)


#################################################################
################# McNemars Test #################################
#################################################################
allCorrectIncorrectResults <- bind_cols(
  results_XGB[[1]]$correct,
  results_Logistic[[1]]$correct,
  results_NN[[1]]$correct,
  results_NN_deep1[[1]]$correct,
  results_randomForest[[1]]$correct,
  results_SVM[[1]]$correct,
  results_SVM_linear[[1]]$correct,
  results_lightGBM[[1]]$correct
) %>% 
  setNames(c("XGB", "Logistic", "NN", "NNDeep", "RandomForest", "SVM", "SVMLinear", "LightGBM"))


mcNemarPvalues <- matrix(NA, nrow = ncol(allCorrectIncorrectResults),ncol = ncol(allCorrectIncorrectResults))
mcNemarPvalues[lower.tri(mcNemarPvalues)] <- combn(allCorrectIncorrectResults, 2, function(x){
  mcnemar.test(x[[1]],
               x[[2]],
               correct = TRUE)$p.value}, simplify = T)

mcNemarPvalues[upper.tri(mcNemarPvalues)] <- t(mcNemarPvalues)[upper.tri(t(mcNemarPvalues))]
mcNemarPvalues

colnames(mcNemarPvalues) <- colnames(allCorrectIncorrectResults)
rownames(mcNemarPvalues) <- colnames(allCorrectIncorrectResults)
mcNemarPvalues

mcNemarPvalues %>% 
  stargazer(digits = 5, digits.extra = 1, rownames = TRUE, colnames = TRUE,
            title = "McNemar P-values for model combinations")

#################################################################
#################################################################
#################################################################

#################################################################
################# Cochrans Q test ###############################
#################################################################
library(nonpar)

allCorrectIncorrectResults %>% 
  mutate_if(is.factor, as.numeric) %>%
  mutate(across(where(is.numeric), list(func = function(x){x-1}))) %>% 
  select(contains("func")) %>% 
  #select(contains(c("Logistic", "NN", "SVM"))) %>% 
  cochrans.q()


#################################################################
#################################################################
#################################################################