rm(list=ls())
options(scipen=999)

# For keras/ tensor flow : Need to use: (added 21_09_2020)

#keras::use_python("/home/msmith/anaconda3/bin/python3.6", required = T)

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
library(magrittr)
library(patchwork)
library(tidyverse)
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


###########################################################

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
#################################################################
set.seed(1234)

full_data <- data %>% 
  future_map(., ~as_tibble(.) %>%
               rename(ID = X) %>% 
               sample_n(nrow(.)))

train <- full_data %>% 
  future_map(., ~as_tibble(.) %>% 
               sample_frac(0.75))

test <- full_data %>% 
  future_map2(.x = ., 
              .y = train, 
              .f = ~anti_join(.x, .y, by = "ID"))



#################################################################
################# Logistic Regression ###########################
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

X_test_scaled <- test_scaled %>%
  map(., ~dplyr::select(., -ID, -Postcode, -City, -Latitude, -Longitude, -Region.in.country, 
                        -Major.sectors, -NACE.Rev..2.main.section, -BvD.ID.number, -status) %>% 
        as.matrix())



#################################################################

XY_vars <- train_scaled[[1]] %>% 
  dplyr::select(-ID, -Postcode, -City, -Latitude, -Longitude, -Region.in.country,
                -Major.sectors, -NACE.Rev..2.main.section, -BvD.ID.number)

X_formula <- str_subset(names(XY_vars), "status", negate = TRUE)
FORMULA <- reformulate(X_formula, response = "status")

#################### Stepwise Logistic Regression ################

library(MASS)
priorToProcess <- 1 # i.e. select which list element 1, 2, 3, 4

# Full Logistic Regression
logistModel <- glm(FORMULA, data = train_scaled[[priorToProcess]], family = "binomial")
logistModel %>% coef()

# Stepwise Logistic Regression
modelStepwise <- logistModel %>% 
  stepAIC(trace = FALSE)
modelStepwise %>% coef()
# Chose a final model in which one variable has been removed...
# setdiff(
#   names(logistModel %>% coef()),
#   names(modelStepwise %>% coef())
# )

# Compare the full and stepwise model
# Make predictions

observedClasses <- test_scaled[[priorToProcess]]$status
#Logistic
logisticProbs <- predict(logistModel, test_scaled[[priorToProcess]], type = 'response')
logisticPredictedClasses <- ifelse(logisticProbs > 0.5, 1, 0)
mean(logisticPredictedClasses == observedClasses)
# Stepwise
stepwiseProbs <- predict(modelStepwise, test_scaled[[priorToProcess]], type = 'response')
stepwisePredictedClasses <- ifelse(stepwiseProbs > 0.5, 1, 0)
mean(stepwisePredictedClasses == observedClasses)
#################################################################

############## Penalised Logistic Regression ####################

####### LASSO ########

library(glmnet)
# alpha: the elasticnet mixing parameter. Allowed values include:
# “1”: for lasso regression
# “0”: for ridge regression
# a value between 0 and 1 (say 0.3) for elastic net regression.
# lamba: a numeric value defining the amount of shrinkage. Should be specify by analyst. Use cv.glmnet to find the best lambda

lassoCV <- cv.glmnet(x = X_train_scaled[[priorToProcess]], y = train_scaled[[priorToProcess]]$status, family = "binomial", alpha = 1)
plot(lassoCV)
lassoCV %>% coef(.$lambda.min) # therefore using 1se produces a more simple model but maybe the predictions are worse
lassoCV %>% coef(.$lambda.1se)

# Using lambda min
lassoModelLambdaMin <- glmnet(x = X_train_scaled[[priorToProcess]], y = train_scaled[[priorToProcess]]$status, family = "binomial", alpha = 1, lambda = lassoCV$lambda.min)
lassoModelLambdaMin %>% coef()

# Make predictions for the Lasso model using lambda min
lassoProbsLambdaMin <- predict(lassoModelLambdaMin, newx = X_test_scaled[[priorToProcess]], type = 'response')
lassoPredictedClassesLambdaMin <- ifelse(lassoProbsLambdaMin > 0.5, 1, 0)
mean(lassoPredictedClassesLambdaMin == observedClasses)


# Using lambda 1se (which is a more simple model)

lassoModelLambda1se <- glmnet(x = X_train_scaled[[priorToProcess]], y = train_scaled[[priorToProcess]]$status, family = "binomial", alpha = 1, lambda = lassoCV$lambda.1se)
lassoModelLambda1se %>% coef()

# Make predictions for the Lasso model using lambda 1se
lassoProbsLambda1se <- predict(lassoModelLambda1se, newx = X_test_scaled[[priorToProcess]], type = 'response')
lassoPredictedClassesLambda1se <- ifelse(lassoProbsLambda1se > 0.5, 1, 0)
mean(lassoPredictedClassesLambda1se == observedClasses)

####### RIDGE ########
ridgeCV <- cv.glmnet(x = X_train_scaled[[priorToProcess]], y = train_scaled[[priorToProcess]]$status, family = "binomial", alpha = 0) # Setting alpha to 0 gives a ridge regression
plot(ridgeCV)
ridgeCV %>% coef(.$lambda.min) # therefore using 1se produces a more simple model but maybe the predictions are worse
ridgeCV %>% coef(.$lambda.1se)

# Using lambda min
ridgeModelLambdaMin <- glmnet(x = X_train_scaled[[priorToProcess]], y = train_scaled[[priorToProcess]]$status, family = "binomial", alpha = 0, lambda = ridgeCV$lambda.min)
ridgeModelLambdaMin %>% coef()

# Make predictions for the Lasso model using lambda min
ridgeProbsLambdaMin <- predict(ridgeModelLambdaMin, newx = X_test_scaled[[priorToProcess]], type = 'response')
ridgePredictedClassesLambdaMin <- ifelse(ridgeProbsLambdaMin > 0.5, 1, 0)
mean(ridgePredictedClassesLambdaMin == observedClasses)

# Using lambda 1se (which is a more simple model)

ridgeModelLambda1se <- glmnet(x = X_train_scaled[[priorToProcess]], y = train_scaled[[priorToProcess]]$status, family = "binomial", alpha = 0, lambda = ridgeCV$lambda.1se)
ridgeModelLambda1se %>% coef()

# Make predictions for the Lasso model using lambda 1se
ridgeProbsLambda1se <- predict(ridgeModelLambda1se, newx = X_test_scaled[[priorToProcess]], type = 'response')
ridgePredictedClassesLambda1se <- ifelse(ridgeProbsLambda1se > 0.5, 1, 0)
mean(ridgePredictedClassesLambda1se == observedClasses)

####### ELASTIC NET ########

# alphalist <- seq(0 , 1, by = 0.1)
# elasticnet <- lapply(alphalist, function(a){
#   cv.glmnet(x = X_train_scaled[[priorToProcess]], y = train_scaled[[priorToProcess]]$status, alpha = a, family = "binomial", lambda.min.ratio = .001)
# })
# 
# for(i in 1:11){
#   print(paste("minimum cvm: ", round(min(elasticnet[[i]]$cvm), 4), " with corresponding lambda min: ", round(elasticnet[[i]]$lambda.min, 4), " and lambda 1se: ", round(elasticnet[[i]]$lambda.1se, 4)))
#   }

library(doParallel)
a <- seq(0.1, 0.9, 0.1)
search <- foreach(i = a, .combine = rbind) %dopar% {
  cv <- cv.glmnet(x = X_train_scaled[[priorToProcess]], y = train_scaled[[priorToProcess]]$status, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = i)
  data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.1se], lambda.1se = cv$lambda.1se, alpha = i)
}
elasticNetCV <- search[search$cvm == min(search$cvm), ]
elasticNetModel <- glmnet(x = X_train_scaled[[priorToProcess]], y = train_scaled[[priorToProcess]]$status, family = "binomial", lambda = elasticNetCV$lambda.1se, alpha = elasticNetCV$alpha)
coef(elasticNetModel)

# Make predictions for the optimised Elastic Net model using lambda 1se
elasticNetModelProbs <- predict(elasticNetModel, newx = X_test_scaled[[priorToProcess]], type = 'response')
elasticNetModelClasses <- ifelse(elasticNetModelProbs > 0.5, 1, 0)
mean(elasticNetModelClasses == observedClasses)


########################################################
############ Combine everything together ###############
########################################################


library(gtools)
# Process original Logistic model
logisticCoefficients <- coef(summary(logistModel)) %>% 
  data.frame() %>% 
  rownames_to_column("Variable") %>% 
  dplyr::select(-c('Std..Error', 'z.value')) %>% 
  rename(pvalue = "Pr...z..") %>% 
  mutate(
    OrigLogisticEstimate = round(Estimate, 3),
    OrigLogisticpvalue = stars.pval(pvalue)
  ) %>% 
  dplyr::select(-c(Estimate, pvalue))

# Process stepwise Logistic model
stepwiseCoefficients <- coef(summary(modelStepwise)) %>% 
  data.frame() %>% 
  rownames_to_column("Variable") %>% 
  dplyr::select(-c('Std..Error', 'z.value')) %>% 
  rename(pvalue = "Pr...z..") %>% 
  mutate(
    StepwiseLogisticEstimate = round(Estimate, 3),
    StepwiseLogisticpvalue = stars.pval(pvalue)
  ) %>% 
  dplyr::select(-c(Estimate, pvalue))

# Process LASSO lambda min model
lassoModelLambdaMinCoefficients <- coef(lassoModelLambdaMin) %>% 
  as.matrix() %>% 
  data.frame() %>% 
  rownames_to_column("Variable") %>%  
  rename(Estimate = s0) %>% 
  mutate(
    LASSOLambaMinEstimate = round(Estimate, 3)
  ) %>% 
  dplyr::select(-c(Estimate))

# Process LASSO lambda 1se model

lassoModelLambda1seCoefficients <- coef(lassoModelLambda1se) %>% 
  as.matrix() %>%  
  data.frame() %>% 
  rownames_to_column("Variable") %>%  
  rename(Estimate = s0) %>% 
  mutate(
    LASSO1seEstimate = round(Estimate, 3),
    LASSO1seEstimate = ifelse(LASSO1seEstimate == 0.000, ".", LASSO1seEstimate)
  ) %>% 
  dplyr::select(-c(Estimate))


# Process ridge min model
ridgeModelLambdaMinCoefficients <- coef(ridgeModelLambdaMin) %>% 
  as.matrix() %>% 
  data.frame() %>% 
  rownames_to_column("Variable") %>%  
  rename(Estimate = s0) %>% 
  mutate(
    RIDGEMinEstimate = round(Estimate, 3)
  ) %>% 
  dplyr::select(-c(Estimate))

ridgeModelLambda1seCoefficients <- coef(ridgeModelLambda1se) %>% 
  as.matrix() %>%  
  data.frame() %>% 
  rownames_to_column("Variable") %>%  
  rename(Estimate = s0) %>% 
  mutate(
    RIDGE1seEstimate = round(Estimate, 3),
    RIDGE1seEstimate = ifelse(RIDGE1seEstimate == 0.000, ".", RIDGE1seEstimate)
  ) %>% 
  dplyr::select(-c(Estimate))

# Optimised Elastic Net coefficients

elasticNetModelLambda1seCoefficients <- coef(elasticNetModel) %>% 
  as.matrix() %>% 
  data.frame() %>% 
  rownames_to_column("Variable") %>%  
  rename(Estimate = s0) %>% 
  mutate(
    ELASTICNETEstimate = round(Estimate, 3),
    ELASTICNETEstimate = ifelse(ELASTICNETEstimate == 0.000, ".", ELASTICNETEstimate)
  ) %>% 
  dplyr::select(-c(Estimate))

# logisticCoefficients %>% 
#   left_join(stepwiseCoefficients, by = "Variable") %>% 
#   left_join(lassoModelLambdaMinCoefficients, by = "Variable") %>% # Perhaps in LaTeX colour the Estimates based on pvalues and then remove the pvalues columns
#   left_join(lassoModelLambda1seCoefficients, by = "Variable") %>% 
#   left_join(ridgeModelLambdaMinCoefficients, by = "Variable") %>% 
#   left_join(ridgeModelLambda1seCoefficients, by = "Variable") %>% 
#   left_join(elasticNetModelLambda1seCoefficients, by = "Variable")

################### Model LateX summary ###############
stargazer(
  logistModel,
  modelStepwise,
  title = "Variable Selection Regression Results",
  align = TRUE, no.space = TRUE, font.size = "footnotesize",
  dep.var.labels = c("Binary Status of Bankruptcy"),
  column.labels = c(
    "Original Logistic", "Stepwise Logistic")
)

logisticCoefficients %>% dplyr::select(-c(OrigLogisticpvalue)) %>% 
  left_join(stepwiseCoefficients, by = "Variable") %>% dplyr::select(-c(StepwiseLogisticpvalue)) %>% 
  left_join(lassoModelLambdaMinCoefficients, by = "Variable") %>% 
  left_join(lassoModelLambda1seCoefficients, by = "Variable") %>% 
  left_join(ridgeModelLambdaMinCoefficients, by = "Variable") %>% 
  left_join(ridgeModelLambda1seCoefficients, by = "Variable") %>% 
  left_join(elasticNetModelLambda1seCoefficients, by = "Variable") %>% 
  mutate_all(funs(str_remove(., "\\.*"))) %>% 
  setNames(c("Variable", "Logistic", "Stepwise", "LASSOMin", "LASSO1se", "RIDGEMin", "RIDGE1se", "ELASTICNET1se")) %>% 
  stargazer(
    summary = FALSE,
    rownames = FALSE,
    title = "Variable Selection Regression Results",
    align = TRUE, no.space = TRUE, font.size = "footnotesize"
  )

############# Model performances ###############
# mean(logisticPredictedClasses == observedClasses)
# mean(stepwisePredictedClasses == observedClasses)
# mean(lassoPredictedClassesLambdaMin == observedClasses)
# mean(lassoPredictedClassesLambda1se == observedClasses)
# mean(ridgePredictedClassesLambdaMin == observedClasses)
# mean(ridgePredictedClassesLambda1se == observedClasses)
# mean(elasticNetModelClasses == observedClasses)

conMat_LogisticVarSelection <- confusionMatrix(factor(logisticPredictedClasses), factor(observedClasses))
conMat_StepwiseVarSelection <- confusionMatrix(factor(stepwisePredictedClasses), factor(observedClasses))
conMat_LASSOMinVarSelection <- confusionMatrix(factor(lassoPredictedClassesLambdaMin), factor(observedClasses))
conMat_LASSO1seVarSelection <- confusionMatrix(factor(lassoPredictedClassesLambda1se), factor(observedClasses))
conMat_RIDGEMinVarSelection <- confusionMatrix(factor(ridgePredictedClassesLambdaMin), factor(observedClasses))
conMat_RIDGE1seVarSelection <- confusionMatrix(factor(ridgePredictedClassesLambda1se), factor(observedClasses))
conMat_ELASTICNETVarSelection <- confusionMatrix(factor(elasticNetModelClasses), factor(observedClasses))

data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1"),
  Logistic = c(
    round(conMat_LogisticVarSelection$overall[[1]], 3),
    round(conMat_LogisticVarSelection$byClass[[1]], 3),
    round(conMat_LogisticVarSelection$byClass[[2]], 3),
    round(conMat_LogisticVarSelection$byClass[[5]], 3),
    round(conMat_LogisticVarSelection$byClass[[7]], 3)
  ),
  Stepwise = c(
    round(conMat_StepwiseVarSelection$overall[[1]], 3),
    round(conMat_StepwiseVarSelection$byClass[[1]], 3),
    round(conMat_StepwiseVarSelection$byClass[[2]], 3),
    round(conMat_StepwiseVarSelection$byClass[[5]], 3),
    round(conMat_StepwiseVarSelection$byClass[[7]], 3)
  ),
  LASSOMin = c(
    round(conMat_LASSOMinVarSelection$overall[[1]], 3),
    round(conMat_LASSOMinVarSelection$byClass[[1]], 3),
    round(conMat_LASSOMinVarSelection$byClass[[2]], 3),
    round(conMat_LASSOMinVarSelection$byClass[[5]], 3),
    round(conMat_LASSOMinVarSelection$byClass[[7]], 3)
  ),
  LASSO1se = c(
    round(conMat_LASSO1seVarSelection$overall[[1]], 3),
    round(conMat_LASSO1seVarSelection$byClass[[1]], 3),
    round(conMat_LASSO1seVarSelection$byClass[[2]], 3),
    round(conMat_LASSO1seVarSelection$byClass[[5]], 3),
    round(conMat_LASSO1seVarSelection$byClass[[7]], 3)
  ),
  RIDGEMin = c(
    round(conMat_RIDGEMinVarSelection$overall[[1]], 3),
    round(conMat_RIDGEMinVarSelection$byClass[[1]], 3),
    round(conMat_RIDGEMinVarSelection$byClass[[2]], 3),
    round(conMat_RIDGEMinVarSelection$byClass[[5]], 3),
    round(conMat_RIDGEMinVarSelection$byClass[[7]], 3)
  ),
  RIDGE1se = c(
    round(conMat_RIDGE1seVarSelection$overall[[1]], 3),
    round(conMat_RIDGE1seVarSelection$byClass[[1]], 3),
    round(conMat_RIDGE1seVarSelection$byClass[[2]], 3),
    round(conMat_RIDGE1seVarSelection$byClass[[5]], 3),
    round(conMat_RIDGE1seVarSelection$byClass[[7]], 3)
  ),
  ElasticNet = c(
    round(conMat_ELASTICNETVarSelection$overall[[1]], 3),
    round(conMat_ELASTICNETVarSelection$byClass[[1]], 3),
    round(conMat_ELASTICNETVarSelection$byClass[[2]], 3),
    round(conMat_ELASTICNETVarSelection$byClass[[5]], 3),
    round(conMat_ELASTICNETVarSelection$byClass[[7]], 3)
  )
) %>% 
  stargazer(
    summary = FALSE,
    rownames = FALSE,
    title = "Variable Selection Confusion Matrix Results (two years prior)",               # NOTE: This needs changing depening on the year
    align = TRUE, no.space = TRUE, font.size = "footnotesize"
  )


