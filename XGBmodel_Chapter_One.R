#rm(list=ls(all=TRUE))

library(readr)
library(xgboost)

setwd('C:/Users/Matt/Desktop/PhD-thesis/FINAL/')


df <- read_csv("C:/Read_Data")

df$X1 <- NULL

set.seed(777)
xgbdata <- df[sample(nrow(df)),] #Randomly sample the data


library(dplyr)


xgbdata <- xgbdata %>%
  select(-Major.sectors, -NACE.Rev..2.main.section, -NACEcode, -Major.sectors.id, -Region.in.country.id, -Region.in.country)

#Split data between train and test sample sizes

smp_size <- floor(0.75 * nrow(xgbdata))
train_ind <- sample(seq_len(nrow(xgbdata)), size = smp_size)
data_train <- xgbdata[train_ind, ]
data_test <- xgbdata[-train_ind, ]
ids <- sample(nrow(data_train))

x_train <- data_train %>%
  select(-status, -BvD.ID.number)
x_test  <- data_test %>%
  select(-status, -BvD.ID.number)

y_train <- data_train$status
y_test <- data_test$status


sumpos_bankrupt_cases <- sum(y_train == 1)
sumneg_active_cases <- sum(y_train == 0)


dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train, missing = "NaN")
dtest <- xgb.DMatrix(data = as.matrix(x_test), label = y_test, missing = "NaN")


##### Cross Validation
#####################################################################################################
#####################################################################################################
#####################################################################################################

searchGridSubCol <- expand.grid(subsample = c(0.75, 1), #Range (0,1], default = 1, set to 0.5 will prevent overfitting
                                colsample_bytree = c(0.75, 1), #Range (0,1], default = 1
                                max_depth = c(3, 5, 8), #Range (0, inf], default = 6 ## Tree depth
                                min_child = c(1, 5, 10), #Range (0, inf], default = 1
                                eta = c(0.1, 0.05, 1), #Range (0,1], default = 0.3, step size shrinkage
                                gamma = c(0, 0.5, 1, 1.5), #Range (0, inf], default = 0, minimum loss reduction required to make a further partition
                                lambda = c(1), #Default = 1, L2 regularisation on weights, higher the more conservative
                                alpha = c(0), #Default = 0, L1 regularisation on weights, higher the more conservative
                                max_delta_step = c(0), #Range (0, inf], default = 0 (Helpful for logisitc regression when class is extremely imbalanced, set to value 1-10 may help control the update)
                                colsample_bylevel = c(1) #Range (0,1], default = 1
)

ntrees <- 100 #this should be nrounds, also since we have early_stopping_round this number should be very high 1500

nfold <- 10 #Number of CV folds

watchlist <- list(train=dtrain, test=dtest)

#Take the grid search defined previously and store evaluation results
system.time(
  AUCHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
    #Extract Parameters to test
    currentSubsampleRate <- parameterList[["subsample"]]
    currentColsampleRate <- parameterList[["colsample_bytree"]]
    currentDepth <- parameterList[["max_depth"]]
    currentEta <- parameterList[["eta"]]
    currentMinChild <- parameterList[["min_child"]]
    gamma <- parameterList[["gamma"]]
    lambda <- parameterList[["lambda"]]
    alpha <- parameterList[["alpha"]]
    max_delta_step <- parameterList[["max_delta_step"]]
    colsample_bylevel <- parameterList[["colsample_bylevel"]]
    xgboostModelCV <- xgb.cv(data =  dtrain, 
                             nrounds = ntrees, 
                             nfold = nfold, 
                             showsd = TRUE,
                             "scale_pos_weight" = sumneg_active_cases /  sumpos_bankrupt_cases,
                             metrics = c("auc", "logloss", "error"),
                             verbose = TRUE, 
                             "eval_metric" = c("auc", "logloss", "error"),
                             "objective" = "binary:logistic", #Outputs a probability "binary:logitraw" - outputs score before logistic transformation
                             "max.depth" = currentDepth, 
                             "eta" = currentEta,
                             "gamma" = gamma,
                             "lambda" = lambda,
                             "alpha" = alpha,
                             "subsample" = currentSubsampleRate, 
                             "colsample_bytree" = currentColsampleRate,
                             print_every_n = 1,
                             "min_child_weight" = currentMinChild,
                             booster = "gbtree", #booster = "dart"  #using dart can help improve accuracy aparantly.
                             #early_stopping_rounds = 10,
                             watchlist = watchlist,
                             seed = 1234)
    xvalidationScores <<- as.data.frame(xgboostModelCV$evaluation_log)
    train_auc_mean <- tail(xvalidationScores$train_auc_mean, 1)
    test_auc_mean <- tail(xvalidationScores$test_auc_mean, 1)
    train_logloss_mean <- tail(xvalidationScores$train_logloss_mean, 1)
    test_logloss_mean <- tail(xvalidationScores$test_logloss_mean, 1)
    train_error_mean <- tail(xvalidationScores$train_error_mean, 1)
    test_error_mean <- tail(xvalidationScores$test_error_mean, 1)
    output <- return(c(train_auc_mean, test_auc_mean, train_logloss_mean, test_logloss_mean, train_error_mean, test_error_mean, xvalidationScores, currentSubsampleRate, currentColsampleRate, currentDepth, currentEta, gamma, lambda, alpha, max_delta_step, colsample_bylevel, currentMinChild))
    hypemeans <- which.max(AUCHyperparameters[[1]]$test_auc_mean)
    output2 <- return(hypemeans)
  }))

#save(AUCHyperparameters, file="AUCHyperparameters_LARGE_list.RData")
#save.image("C:/Users/Matt/Desktop/PhD-thesis/FINAL/AUCHyperparameters_saved_up_to_output.RData")

output <- as.data.frame(t(sapply(AUCHyperparameters, '[', c(1:6, 20:29))))
varnames <- c("TrainAUC", "TestAUC", "TrainLogloss", "TestLogloss", "TrainError", "TestError", "SubSampRate", "ColSampRate", "Depth", "eta", "gamma", "lambda", "alpha", "max_delta_step", "col_sample_bylevel", "currentMinChild")
colnames(output) <- varnames

library(data.table)
data.table(output)


output_save <- as.data.frame(lapply(output, unlist))

library(dplyr)

output_save <- output_save %>% 
  arrange(TestAUC, TrainAUC)
output_save <- output_save[dim(output_save)[1]:1,]
data.table(output_save)

output_save <- output_save %>%
  select(-TrainLogloss, -TestLogloss, -TrainError, -TestError, -col_sample_bylevel, -max_delta_step, -lambda, -alpha)

# output_save <- apply(output_save, 2, as.character)
# write.table(output_save, file = "output_flip.csv", sep = ",", quote = FALSE, row.names = F)


#Creating the evaluation plots
#####################################################################################################
########################################### AUC ~####################################################
#####################################################################################################

############################ Train and Test plots

# Plot the training error AUC metric for all CV validation

result_train_auc <- NULL
for(i in 1:length(AUCHyperparameters)) {
  temp <- AUCHyperparameters[[i]]$train_auc_mean
  result_train_auc$newcol[[i]] <- temp
}


# Plot the testing error AUC metric for all CV validation
result_test_auc <- NULL
for(i in 1:length(AUCHyperparameters)) {
  temp <- AUCHyperparameters[[i]]$test_auc_mean
  result_test_auc$newcol[[i]] <- temp
}

library(tidyr)

n <- length(result_train_auc[[1]])

data_1 <- tibble(group_id = 1:n, value = result_train_auc[[1]], ID = rep(list(1:100)))
data_1 <- unnest(data_1)

data_2 <- tibble(group_id = 1:n, value = result_test_auc[[1]], ID = rep(list(1:100)))
data_2 <- unnest(data_2)

series_order  <- order(filter(data_1, ID ==  100)$value)

combo_data <- bind_rows(Train = data_1, Test = data_2, .id = "type") %>%
  mutate(group_id = factor(group_id, levels = series_order))

combo_data$type <- factor(combo_data$type, levels=c("Train","Test"))

combo_data %>%
  filter(type == "Test") %>%
  filter(value == max(value))

library(ggplot2)

#Get the max test AUC score 81, and the best performing group_id 354
combo_data %>%
  filter(type == "Test") %>%
  filter(value == max(value))

#Get the worst performing model number, group_id 191
combo_data %>%
  filter(type == "Test") %>%
  filter(ID >= 25) %>%
  filter(value == min(value))

Train_Test_AUC <- combo_data %>%
  ggplot(aes(ID)) + 
  geom_path(aes(y = value, group = group_id), colour = "grey50", alpha = 0.1) +
  facet_wrap("type", scales = "free") +
  geom_path(data = combo_data %>%
              filter(group_id == 354), aes(y = value), colour = "green", size = 0.725) +
  geom_path(data = combo_data %>%
              filter(group_id == 191), aes(y = value), colour = "red2", size = 0.725) +
  geom_point(data = combo_data %>%
               filter(type == "Test") %>%
               filter(group_id == 354) %>%
               filter(ID == 81), aes(y = value), colour = "red", size = 1.5) +
  geom_text(data = combo_data %>%
              filter(type == "Test") %>%
              filter(group_id == 354) %>%
              filter(ID == 80), aes(x = 88, y = 0.916, label = "88 trees")) +
  labs(title = "Train and Test AUC curves", y = "Average AUC scores across k-folds", x = "Number of trees") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none")



#Obtain the best iteration
best_iteration <- combo_data %>%
  filter(type == "Test") %>%
  group_by(group_id) %>%
  filter(value == max(value)) %>%
  arrange(value) #best iteration at the end of the df


# best_iteration <- data_2 %>%
#   group_by(group_id) %>%
#   filter(group_id == 354) #Model 354 had the best_iteration at iter 81

#n = 4; tiff("Train_Test_AUC.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(Train_Test_AUC); dev.off()

###################################################################################################################
################################################# Log Loss ########################################################
###################################################################################################################

# Plot the training logloss metric for all CV validation
result_train_Logloss <- NULL
for(i in 1:length(AUCHyperparameters)) {
  temp <- AUCHyperparameters[[i]]$train_logloss_mean
  result_train_Logloss$newcol[[i]] <- temp
}

# Plot the testing logloss metric for all CV validation
result_test_Logloss <- NULL
for(i in 1:length(AUCHyperparameters)) {
  temp <- AUCHyperparameters[[i]]$test_logloss_mean
  result_test_Logloss$newcol[[i]] <- temp
}

n <- length(result_train_Logloss[[1]])

data_1 <- tibble(group_id = 1:n, value = result_train_Logloss[[1]], ID = rep(list(1:100)))
data_1 <- unnest(data_1)
data_2 <- tibble(group_id = 1:n, value = result_test_Logloss[[1]], ID = rep(list(1:100)))
data_2 <- unnest(data_2)


series_order  <- order(filter(data_1, ID ==  100)$value)

combo_data <- bind_rows(Train = data_1, Test = data_2, .id = "type") %>%
  mutate(group_id = factor(group_id, levels = series_order))

combo_data$type <- factor(combo_data$type, levels=c("Train","Test"))

Train_Test_Logloss <- combo_data %>%
  ggplot(aes(ID)) + 
  geom_path(aes(y = value, group = group_id), colour = "grey50", alpha = 0.1) +
  facet_wrap("type", scales = "free") +
  geom_path(data = combo_data %>%
              filter(group_id == 354), aes(y = value), colour = "deepskyblue4", size = 0.725) +
  geom_path(data = combo_data %>%
              filter(group_id == 191), aes(y = value), colour = "red2", size = 0.725) +
  labs(title = "Train and Test Logloss curves", y = "Average Logloss scores across k-folds", x = "Number of trees") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none")


#n = 4; tiff("Train_Test_Logloss.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(Train_Test_Logloss); dev.off()

###################################################################################################################
#######################################################Train Error ################################################
###################################################################################################################


# Plot the training TrainError metric for all CV validation

result_train_Error <- NULL
for(i in 1:length(AUCHyperparameters)) {
  temp <- AUCHyperparameters[[i]]$train_error_mean
  result_train_Error$newcol[[i]] <- temp
}

# Plot the testing TestError metric for all CV validation

result_test_Error <- NULL
for(i in 1:length(AUCHyperparameters)) {
  temp <- AUCHyperparameters[[i]]$test_error_mean
  result_test_Error$newcol[[i]] <- temp
}

n <- length(result_train_Error[[1]])

data_1 <- tibble(group_id = 1:n, value = result_train_Error[[1]], ID = rep(list(1:100)))
data_1 <- unnest(data_1)
data_2 <- tibble(group_id = 1:n, value = result_test_Error[[1]], ID = rep(list(1:100)))
data_2 <- unnest(data_2)

# To order by final value
series_order  <- order(filter(data_1, ID ==  100)$value)

combo_data <- bind_rows(Train = data_1, Test = data_2, .id = "type") %>%
  mutate(group_id = factor(group_id, levels = series_order))

combo_data$type <- factor(combo_data$type, levels=c("Train","Test"))

Train_Test_Error <- combo_data %>%
  ggplot(aes(ID)) + 
  geom_path(aes(y = value, group = group_id), colour = "grey50", alpha = 0.1) +
  facet_wrap("type", scales = "free") +
  geom_path(data = combo_data %>%
              filter(group_id == 354), aes(y = value), colour = "deepskyblue4", size = 0.725) +
  geom_path(data = combo_data %>%
              filter(group_id == 191), aes(y = value), colour = "red2", size = 0.725) +
  labs(title = "Train and Test Error curves", y = "Average Error scores across k-folds", x = "Number of trees") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none")



#n = 4; tiff("Train_Test_Error.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(Train_Test_Error); dev.off()


#####################################################################################################
#####################################################################################################
#####################################################################################################

NOTE: That the two models below are modeling the default model and the optimised model.

#######################################################################################################
#####################################Finding the optimal ntrees########################################
#######################################################################################################

library(xgboost)
dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = as.numeric(y_train), missing = "NaN")
dtest <- xgb.DMatrix(data = as.matrix(x_test), label = as.numeric(y_test), missing = "NaN")

#Finding ntrees using the optimal parameters

gc()

library(caret)

cv <- createFolds(y_train, k = 10)
params_ntrees <- list("eta" = 0.1,
                      "max_depth" = 5,
                      "colsample_bytree" = 0.75,
                      "min_child_weight" = 5,
                      "subsample"= 1,
                      "objective"="binary:logistic",
                      "gamma" = 0.5,
                      "lambda" = 1, #Default
                      "alpha" = 0, #Default
                      "max_delta_step" = 0, #Default
                      "colsample_bylevel" = 1, #Default
                      "eval_metric"= "auc",
                      "scale_pos_weight" = sumneg_active_cases /  sumpos_bankrupt_cases,
                      "set.seed" = 1234)

xgboost.cv <- xgb.cv(param = params_ntrees, data = dtrain, folds = cv, nrounds = 1500, early_stopping_rounds = 500, metrics = 'auc')

xgboost.cv$best_iteration

# Plot the optimal number of trees for the best model

optimal_ntrees <- ggplot(xgboost.cv$evaluation_log, aes(x = iter, y = test_auc_mean)) + 
  geom_line() +
  geom_point(data = xgboost.cv$evaluation_log[88], aes(x=iter, y=test_auc_mean), colour="red", size=2) +
  labs(title = "Optimal Number of Trees", y = "Average AUC score across k-folds", x = "Number of trees") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none") +
  geom_text(data = xgboost.cv$evaluation_log %>%
              filter(iter == 88), aes(x = 88, y = 0.916, label = "88 trees")) +
  geom_errorbar(aes(ymin=test_auc_mean-test_auc_std, ymax=test_auc_mean+test_auc_std), width=.1, colour = "grey", alpha = 0.2, linetype = "solid")


#n = 4; tiff("Optimal_ntrees.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(optimal_ntrees); dev.off()

################################################### Running the XGBoost model

############ Firstly run a base XGBoost model not optimised, with default parameters

watchlist <- list("train" = dtrain)
nround <- xgboost.cv$best_iteration
params_base <- list("eta" = 0.3,
                    "max_depth" = 6,
                    "colsample_bytree" = 1,
                    "min_child_weight" = 1,
                    "subsample"= 1,
                    "objective"="binary:logistic",
                    "gamma" = 0,
                    "lambda" = 1, #Default
                    "alpha" = 0, #Default
                    "max_delta_step" = 0, #Default
                    "colsample_bylevel" = 1, #Default
                    "eval_metric"= "auc",
                    "scale_pos_weight" = sumneg_active_cases /  sumpos_bankrupt_cases,
                    "set.seed" = 1234)

model_base <- xgb.train(params_base, dtrain, nround, watchlist)
pred_base <- predict(model_base, dtest, type = 'prob')

results_base <- NULL
results_base$pred <- pred_base
results_base$prediction <- ifelse(pred_base > 0.50, 1, 0)
results_base$testactual <- y_test
results$BvD.ID.number <- data_test$BvD.ID.number
results_base <- as.data.frame(results_base)

library(caret)

conMat_base <- confusionMatrix(as.factor(results_base$prediction), as.factor(results_base$testactual), mode = "everything")
conMat_base
conMat_base$table

#Overall model variable importance

importancexgb_base <- xgb.importance(colnames(x_train), model = model_base)
importancexgb_base_plot <- xgb.ggplot.importance(importancexgb_base, top_n = 10, n_clusters = 1) +
  labs(title = "Base Model Variable Importance", y = "Top 10 variables", x = "Importance Gain") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none")

#n = 4; tiff("importancexgb_base_plot.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(importancexgb_base_plot); dev.off()

########## Plotting the ROC curve

library(pROC)

roc_base <- roc(results_base$testactual, results_base$pred)
gg_roc_base <- ggroc(roc_base) +
  labs(title = "Base Model ROC curve", y = "Sensitivity", x = "Specificity") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1)


#n = 4; tiff("roc_base.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(gg_roc_base); dev.off()

################# END of base XGBoost model not optimised

################ BEGIN XGBoost model optimised

watchlist <- list("train" = dtrain)

params <- list("eta" = 0.1,
               "max_depth" = 5,
               "colsample_bytree" = 0.75,
               "min_child_weight" = 5,
               "subsample"= 1,
               "objective"="binary:logistic",
               "gamma" = 0.5,
               "lambda" = 1, #Default
               "alpha" = 0, #Default
               "max_delta_step" = 0, #Default
               "colsample_bylevel" = 1, #Default
               "eval_metric"= "auc",
               "scale_pos_weight" = sumneg_active_cases /  sumpos_bankrupt_cases,
               "set.seed" = 2918736852786
)

nround <- xgboost.cv$best_iteration

model <- xgb.train(params, dtrain, nround, watchlist)
pred <- predict(model, dtest, type = 'prob')

##########

results <- NULL
results$pred <- pred
results$prediction <- ifelse(pred > 0.5, 1, 0)
results$testactual <- y_test
results$BvD.ID.number <- data_test$BvD.ID.number
results <- as.data.frame(results)

##########

library(caret)
conMat<- confusionMatrix(as.factor(results$prediction), as.factor(results$testactual), mode = "everything")
conMat
conMat$table

#Overall model variable importance

importancexgb <- xgb.importance(colnames(x_train), model = model)
importancexgb_plot <- xgb.ggplot.importance(importancexgb, top_n = 10, n_clusters = 1) +
  labs(title = "Optimised Model Variable Importance", y = "Top 10 variables", x = "Importance Gain") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none")

#n = 4; tiff("importancexgb_optimised_plot.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(importancexgb_plot); dev.off()

################################### Plotting side by side Base model and Optimised model feature importance
require(gridExtra)
importancexgb_grid <- grid.arrange(importancexgb_base_plot, importancexgb_plot, ncol=2)
################################### END plotting the auc scores

roc_optimised <- roc(results$testactual, results$pred)
gg_roc_optimised <- ggroc(roc_optimised) +
  labs(title = "Optimised Model ROC curve", y = "Sensitivity", x = "Specificity") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1)


#n = 4; tiff("roc_optimised.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(gg_roc_optimised); dev.off()


################################################################################################################
#Plot both the base and optimised ROC curves

roc_base_opt <- ggroc(list("ROC Base" = roc_base, "ROC Optimised" = roc_optimised), legacy.axes = TRUE, linetype = 1, colour = "grey", alpha = 0.5) +
  geom_line(size = 1, alpha = 0.7) +
  scale_colour_discrete(name = "Model") +
  labs(title= "ROC curve", 
       y = "True Positive Rate (Sensitivity)",
       x = "False Positive Rate (1-Specificity)") +
  theme_bw(base_size = 11, base_family = "") +
  #theme(aspect.ratio = 1) + #Plot the AUC of both models
  geom_abline(show.legend = TRUE, alpha = 0.7) +
  geom_text(aes(x = 0.55, y = 0.25, label = paste("\n ", "Base", "\n Optimised")), size = 3, colour = "grey") +
  geom_text(aes(x = 0.65, y = 0.25, label = paste("AUC", "\n ", round(roc_base$auc, 3), "\n", round(roc_optimised$auc, 3))), size = 3, colour = "grey") +
  geom_text(aes(x = 0.75, y = 0.25, label = paste("Sens", "\n ", round(conMat_base$byClass[[1]], 3), "\n", round(conMat$byClass[[1]], 3))), size = 3, colour = "grey") +
  geom_text(aes(x = 0.85, y = 0.25, label = paste("Spec", "\n ", round(conMat_base$byClass[[2]], 3), "\n", round(conMat$byClass[[2]], 3))), size = 3, colour = "grey")
#geom_text(aes(x = 0.85, y = 0.25, label = paste("Base AUC", round(roc_base$auc, 3), "\n Optimised AUC", round(roc_optimised$auc, 3), "\n F1 Score Base", round(conMat_base$byClass[[7]], 3), "\n F1 Score Optimised", round(conMat$byClass[[7]], 3))), size = 3, colour = "grey") #This works but not in the form I like, also not sure if correct AUC scores


#n = 4; tiff("roc_base_opt.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(roc_base_opt); dev.off()
# 
# library(Epi)
# ROC(results$pred, results$testactual)
#####################################################################################################

#Plot the density plots for the probabilities from the optimised model


ggplot() + 
  geom_density(data = results %>%
                 filter(testactual == 0), aes(pred), color='blue') + 
  geom_density(data = results %>%
                 filter(testactual == 1), aes(pred), color='red')


results %>%
  filter(testactual == 1) %>%
  filter(prediction == 1) %>%
  summarise(predi = sum(prediction))

sum(results$prediction == 1)
sum(results$testactual == 1)


#write.table(results, file = "stack.csv", sep = ",", quote = FALSE, row.names = F)

##################################### XGBoost Explainer on the Optimised model ##################################

#The XGBoost Explainer

library(xgboostExplainer)

explainer <- buildExplainer(model, dtrain, type = "binary",
                            base_score = 0.5, n_first_tree = model$best_ntreelimit - 1) #Note: nround should be the best_ntreelimit obtained from best_iteration from earlystoppingrounds in xgb.cv

pred.breakdown <- explainPredictions(model, explainer, dtest) # Not used in showwaterfall() command

cat('Breakdown Complete', '\n')
weights = rowSums(pred.breakdown)
pred.xgb = 1 / (1 + exp(-weights))
cat(max(pred - pred.xgb), '\n')

#Find bankrupt firms which actually went bankrupt

find_bankrupt_firms <- as.data.frame(y_test)
find_bankrupt_firms$prediction <- results$prediction
find_bankrupt_firms$BvD.ID.number <- data_test$BvD.ID.number
find_bankrupt_firms$index <- seq.int(nrow(find_bankrupt_firms))

bankrupt_firms <- find_bankrupt_firms %>%
  filter(y_test == 0) %>%
  filter(prediction == 1)
bankrupt_firms$ID <- seq.int(nrow(bankrupt_firms))

###

#search for firms
idx_to_get <- as.integer(sample(bankrupt_firms$index, 1))
y_test[idx_to_get]

showWaterfall(model, explainer, dtest,
              data.matrix(x_test), idx_to_get, type = "binary")

bankrupt_firms %>%
  filter(index == idx_to_get)


#save that firm
idx_to_get <- as.integer(17398) #Bankrupt firm with firm ID 17398
y_test[idx_to_get]

bnkfirm_1 <- showWaterfall(model, explainer, dtest,
                           data.matrix(x_test), idx_to_get, type = "binary")

#n = 4; tiff("bnkfirm_2.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(bnkfirm_1); dev.off()

# Do the same but now for active firms

active_firms <- find_bankrupt_firms %>%
  filter(y_test == 0) %>%
  filter(prediction == 0)
active_firms$ID <- seq.int(nrow(active_firms))

#search for firms
idx_to_get <- as.integer(sample(active_firms$index, 1))
y_test[idx_to_get]

showWaterfall(model, explainer, dtest,
              data.matrix(x_test), idx_to_get, type = "binary")


idx_to_get <- as.integer(9077) #Active firm with firm ID 9077
y_test[idx_to_get]

active_firm_2 <- showWaterfall(model, explainer, dtest,
                               data.matrix(x_test), idx_to_get, type = "binary")

#n = 4; tiff("active_firm_2.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(active_firm_2); dev.off()


# Do the same but now for grey area firms

greyarea_firms <- find_bankrupt_firms %>%
  filter(y_test == 1) %>%
  filter(prediction == 0)
greyarea_firms$ID <- seq.int(nrow(greyarea_firms))

#search for firms
idx_to_get <- as.integer(sample(greyarea_firms$index, 1)) #19112
y_test[idx_to_get]

showWaterfall(model, explainer, dtest,
              data.matrix(x_test), idx_to_get, type = "binary")


idx_to_get <- as.integer(23334) #Grey firm with firm ID 23334
y_test[idx_to_get]

in_the_middle_firm_4 <- showWaterfall(model, explainer, dtest,
                                      data.matrix(x_test), idx_to_get, type = "binary")

#n = 4; tiff("in_the_middle_firm_4.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(in_the_middle_firm_4); dev.off()


############################################### Precision - recall plots

fg <- results %>%
  filter(testactual == 1) %>%
  select(pred)

bg <- results %>%
  filter(testactual == 0) %>%
  select(pred)

library(PRROC)
pr <- pr.curve(scores.class0 = fg$pred, scores.class1 = bg$pred, curve = TRUE)
plot(pr)