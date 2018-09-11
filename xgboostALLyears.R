rm(list=ls(all=TRUE))
library(readr)
library(dplyr)
setwd('C:/Users/Matt/Desktop/PhD-thesis/FINAL/')

xgbdata <- read_csv("C:/Users/Matt/Desktop/PhD-thesis/FINAL/data/xgbdata.csv")
xgbdata1 <- read_csv("C:/Users/Matt/Desktop/PhD-thesis/FINAL/data/xgbdata_lag1.csv")
xgbdata2 <- read_csv("C:/Users/Matt/Desktop/PhD-thesis/FINAL/data/xgbdata_lag2.csv")
xgbdata3 <- read_csv("C:/Users/Matt/Desktop/PhD-thesis/FINAL/data/xgbdata_lag3.csv")

xgbdata1$X1 <- NULL; xgbdata2$X1 <- NULL; xgbdata3$X1 <- NULL

###################################### Removing some firms ######################################################
IDs <- xgbdata %>% filter(status == 0) %>% select(BvD.ID.number) %>% sample_n(42000)

xgbdata <- xgbdata %>% filter(!BvD.ID.number %in% IDs$BvD.ID.number)
xgbdata1 <- xgbdata1 %>% filter(!BvD.ID.number %in% IDs$BvD.ID.number)
xgbdata2 <- xgbdata2 %>% filter(!BvD.ID.number %in% IDs$BvD.ID.number)
xgbdata3 <- xgbdata3 %>% filter(!BvD.ID.number %in% IDs$BvD.ID.number)
#################################################################################################################

#################################################################################################################
# xgbdata$status <- ifelse(xgbdata$status == 1, 0, 1)
# xgbdata1$status <- ifelse(xgbdata1$status == 1, 0, 1)
# xgbdata2$status <- ifelse(xgbdata2$status == 1, 0, 1)
# xgbdata3$status <- ifelse(xgbdata3$status == 1, 0, 1)
################################################################################################################

set.seed(2277799) #2277799
xgbdata <- xgbdata[sample(nrow(xgbdata)),]
xgbdata1 <- xgbdata1[sample(nrow(xgbdata1)),]
xgbdata2 <- xgbdata2[sample(nrow(xgbdata2)),]
xgbdata3 <- xgbdata3[sample(nrow(xgbdata3)),]

smp_size <- floor(0.75 * nrow(xgbdata))

################################################### Split between train and test data #############################

train_ind <- sample(seq_len(nrow(xgbdata)), size = smp_size)
data_train <- xgbdata[train_ind, ]
data_test <- xgbdata[-train_ind, ]
ids <- sample(nrow(data_train))

train_ind1 <- sample(seq_len(nrow(xgbdata1)), size = smp_size)
data_train1 <- xgbdata1[train_ind1, ]
data_test1 <- xgbdata1[-train_ind1, ]
ids1 <- sample(nrow(data_train1))

train_ind2 <- sample(seq_len(nrow(xgbdata2)), size = smp_size)
data_train2 <- xgbdata2[train_ind2, ]
data_test2 <- xgbdata2[-train_ind2, ]
ids2 <- sample(nrow(data_train2))

train_ind3 <- sample(seq_len(nrow(xgbdata3)), size = smp_size)
data_train3 <- xgbdata3[train_ind3, ]
data_test3 <- xgbdata3[-train_ind3, ]
ids3 <- sample(nrow(data_train3))

##############################################################################################################

x_train <- data_train %>%
  select(-status, -BvD.ID.number)
x_test  <- data_test %>%
  select(-status, -BvD.ID.number)

x_train1 <- data_train1 %>%
  select(-status, -BvD.ID.number)
x_test1  <- data_test1 %>%
  select(-status, -BvD.ID.number)

x_train2 <- data_train2 %>%
  select(-status, -BvD.ID.number)
x_test2  <- data_test2 %>%
  select(-status, -BvD.ID.number)

x_train3 <- data_train3 %>%
  select(-status, -BvD.ID.number)
x_test3  <- data_test3 %>%
  select(-status, -BvD.ID.number)

###############################################################################################################

y_train <- data_train$status
y_test <- data_test$status

y_train1 <- data_train1$status
y_test1 <- data_test1$status

y_train2 <- data_train2$status
y_test2 <- data_test2$status

y_train3 <- data_train3$status
y_test3 <- data_test3$status

##############################################################################################################

sumneg_active <- sum(y_train == 0)
sumpos_bankrupt <- sum(y_train == 1)

sumneg_active1 <- sum(y_train1 == 0)
sumpos_bankrupt1 <- sum(y_train1 == 1)

sumneg_active2 <- sum(y_train2 == 0)
sumpos_bankrupt2 <- sum(y_train2 == 1)

sumneg_active3 <- sum(y_train3 == 0)
sumpos_bankrupt3 <- sum(y_train3 == 1)


################################################################################################################

library(xgboost)
dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train, missing = "NaN")
dtest <- xgb.DMatrix(data = as.matrix(x_test), label = y_test, missing = "NaN")

dtrain1 <- xgb.DMatrix(data = as.matrix(x_train1), label = y_train1, missing = "NaN")
dtest1 <- xgb.DMatrix(data = as.matrix(x_test1), label = y_test1, missing = "NaN")

dtrain2 <- xgb.DMatrix(data = as.matrix(x_train2), label = y_train2, missing = "NaN")
dtest2 <- xgb.DMatrix(data = as.matrix(x_test2), label = y_test2, missing = "NaN")

dtrain3 <- xgb.DMatrix(data = as.matrix(x_train3), label = y_train3, missing = "NaN")
dtest3 <- xgb.DMatrix(data = as.matrix(x_test3), label = y_test3, missing = "NaN")

#################################################################################################################
######################################## Parameters #############################################################

watchlist <- list("train" = dtrain)
watchlist1 <- list("train" = dtrain1)
watchlist2 <- list("train" = dtrain2)
watchlist3 <- list("train" = dtrain3)

params <- list("eta" = 0.1, "max_depth" = 5, "colsample_bytree" = 0.75, "min_child_weight" = 5, "subsample"= 1,
               "objective"="binary:logistic", "gamma" = 0.5, "lambda" = 1, "alpha" = 0, "max_delta_step" = 0,
               "colsample_bylevel" = 1, "eval_metric"= "auc",
               "scale_pos_weight" = sumneg_active /  sumpos_bankrupt, "set.seed" = 176)

params1 <- list("eta" = 0.1, "max_depth" = 5, "colsample_bytree" = 0.75, "min_child_weight" = 5, "subsample"= 1,
               "objective"="binary:logistic", "gamma" = 0.5, "lambda" = 1, "alpha" = 0, "max_delta_step" = 0,
               "colsample_bylevel" = 1, "eval_metric"= "auc",
               "scale_pos_weight" = sumneg_active1 /  sumpos_bankrupt1, "set.seed" = 176) 

params2 <- list("eta" = 0.1, "max_depth" = 5, "colsample_bytree" = 0.75, "min_child_weight" = 5, "subsample"= 1,
               "objective"="binary:logistic", "gamma" = 0.5, "lambda" = 1, "alpha" = 0, "max_delta_step" = 0,
               "colsample_bylevel" = 1, "eval_metric"= "auc",
               "scale_pos_weight" = sumneg_active2 /  sumpos_bankrupt2, "set.seed" = 176) 

params3 <- list("eta" = 0.1, "max_depth" = 5, "colsample_bytree" = 0.75, "min_child_weight" = 5, "subsample"= 1,
               "objective"="binary:logistic", "gamma" = 0.5, "lambda" = 1, "alpha" = 0, "max_delta_step" = 0,
               "colsample_bylevel" = 1, "eval_metric"= "auc",
               "scale_pos_weight" = sumneg_active3 /  sumpos_bankrupt3, "set.seed" = 176) 

nround <- 88

######################################## Run the Models ##########################################################

model <- xgb.train(params, dtrain, nround, watchlist)
model1 <- xgb.train(params1, dtrain1, nround, watchlist1)
model2 <- xgb.train(params2, dtrain2, nround, watchlist2)
model3 <- xgb.train(params3, dtrain3, nround, watchlist3)

########################################## Make the Predictions ##################################################

pred <- predict(model, dtest, type = 'prob')
pred1 <- predict(model1, dtest1, type = 'prob')
pred2 <- predict(model2, dtest2, type = 'prob')
pred3 <- predict(model3, dtest3, type = 'prob')

##################################################################################################################

results <- NULL
results$pred <- pred
results$prediction <- ifelse(pred > 0.5, 1, 0)
results$testactual <- y_test
results$BvD.ID.number <- data_test$BvD.ID.number
results <- as.data.frame(results)

results1 <- NULL
results1$pred1 <- pred1
results1$prediction1 <- ifelse(pred1 > 0.5, 1, 0)
results1$testactual1 <- y_test1
results1$BvD.ID.number1 <- data_test1$BvD.ID.number
results1 <- as.data.frame(results1)

results2 <- NULL
results2$pred2 <- pred2
results2$prediction2 <- ifelse(pred2 > 0.5, 1, 0)
results2$testactual2 <- y_test2
results2$BvD.ID.number2 <- data_test2$BvD.ID.number
results2 <- as.data.frame(results2)

results3 <- NULL
results3$pred3 <- pred3
results3$prediction3 <- ifelse(pred3 > 0.5, 1, 0)
results3$testactual3 <- y_test3
results3$BvD.ID.number3 <- data_test3$BvD.ID.number
results3 <- as.data.frame(results3)

#################################### Confusion Matrix##########################################################
library(caret)

conMat <- confusionMatrix(as.factor(results$prediction), as.factor(results$testactual), mode = "everything", positive = "1")
conMat1 <- confusionMatrix(as.factor(results1$prediction1), as.factor(results1$testactual1), mode = "everything", positive = "1")
conMat2 <- confusionMatrix(as.factor(results2$prediction2), as.factor(results2$testactual2), mode = "everything", positive = "1")
conMat3 <- confusionMatrix(as.factor(results3$prediction3), as.factor(results3$testactual3), mode = "everything", positive = "1")

conMat; conMat1; conMat2; conMat3

conMat$table; conMat1$table; conMat2$table; conMat3$table

print(paste("Last avail year: Sensitivity=", conMat$byClass[[1]], "Specificity", conMat$byClass[[2]], "Precision", conMat$byClass[[5]]))
print(paste("Lag 1: Sensitivity=", conMat1$byClass[[1]], "Specificity", conMat1$byClass[[2]], "Precision", conMat1$byClass[[5]]))
print(paste("Lag 2: Sensitivity=", conMat2$byClass[[1]], "Specificity", conMat2$byClass[[2]], "Precision", conMat2$byClass[[5]]))
print(paste("Lag 3: Sensitivity=", conMat3$byClass[[1]], "Specificity", conMat3$byClass[[2]], "Precision", conMat3$byClass[[5]]))


##################################### MCC #####################################################################

TP <- as.numeric(conMat$table[[1]])
FP <- as.numeric(conMat$table[[2]])
FN <- as.numeric(conMat$table[[3]])
TN <- as.numeric(conMat$table[[4]])

MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))

TP <- as.numeric(conMat1$table[[1]])
FP <- as.numeric(conMat1$table[[2]])
FN <- as.numeric(conMat1$table[[3]])
TN <- as.numeric(conMat1$table[[4]])

MCC1 <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))

TP <- as.numeric(conMat2$table[[1]])
FP <- as.numeric(conMat2$table[[2]])
FN <- as.numeric(conMat2$table[[3]])
TN <- as.numeric(conMat2$table[[4]])

MCC2 <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))

TP <- as.numeric(conMat3$table[[1]])
FP <- as.numeric(conMat3$table[[2]])
FN <- as.numeric(conMat3$table[[3]])
TN <- as.numeric(conMat3$table[[4]])

MCC3 <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))

MCC; MCC1; MCC2; MCC3

##################################### G-means ##################################################################
G <- sqrt(conMat$byClass[[1]] * conMat$byClass[[2]])
G1 <- sqrt(conMat1$byClass[[1]] * conMat1$byClass[[2]])
G2 <- sqrt(conMat2$byClass[[1]] * conMat2$byClass[[2]])
G3 <- sqrt(conMat3$byClass[[1]] * conMat3$byClass[[2]])

##################################### Likelihood ################################################################

print(paste("Sensitivity=", conMat$byClass[[1]]))
print(paste("Specificity", conMat$byClass[[2]]))


Pplus <- conMat$byClass[[1]] / (1 - conMat$byClass[[2]])
Pneg <- (1 - conMat$byClass[[1]]) / conMat$byClass[[2]]

Pplus1 <- conMat1$byClass[[1]] / (1 - conMat1$byClass[[2]])
Pneg1 <- (1 - conMat1$byClass[[1]]) / conMat1$byClass[[2]]

Pplus2 <- conMat2$byClass[[1]] / (1 - conMat2$byClass[[2]])
Pneg2 <- (1 - conMat2$byClass[[1]]) / conMat2$byClass[[2]]

Pplus3 <- conMat3$byClass[[1]] / (1 - conMat3$byClass[[2]])
Pneg3 <- (1 - conMat3$byClass[[1]]) / conMat3$byClass[[2]]

Pplus; Pplus1; Pplus2; Pplus3
Pneg; Pneg1; Pneg2; Pneg3

##################################### Balanced Accuracy #########################################################

BA <- 0.5*(conMat$byClass[[1]] + conMat$byClass[[2]])
BA1 <- 0.5*(conMat1$byClass[[1]] + conMat1$byClass[[2]]) 
BA2 <- 0.5*(conMat2$byClass[[1]] + conMat2$byClass[[2]])
BA3 <- 0.5*(conMat3$byClass[[1]] + conMat3$byClass[[2]])

##################################### Youden Index ##############################################################

YI <- conMat$byClass[[1]] - (1 - conMat$byClass[[2]])
YI1 <- conMat1$byClass[[1]] - (1 - conMat1$byClass[[2]])
YI2 <- conMat2$byClass[[1]] - (1 - conMat2$byClass[[2]])
YI3 <- conMat3$byClass[[1]] - (1 - conMat3$byClass[[2]])

################################### Precision Recall plots ####################################################
library(PRROC)

fg <- results %>% filter(testactual == 1) %>% select(pred)
bg <- results %>% filter(testactual == 0) %>% select(pred)

fg1 <- results1 %>% filter(testactual1 == 1) %>% select(pred1)
bg1 <- results1 %>% filter(testactual1 == 0) %>% select(pred1)

fg2 <- results2 %>% filter(testactual2 == 1) %>% select(pred2)
bg2 <- results2 %>% filter(testactual2 == 0) %>% select(pred2)

fg3 <- results3 %>% filter(testactual3 == 1) %>% select(pred3)
bg3 <- results3 %>% filter(testactual3 == 0) %>% select(pred3)

pr <- pr.curve(scores.class0 = fg$pred, scores.class1 = bg$pred, curve = TRUE)
pr1 <- pr.curve(scores.class0 = fg1$pred1, scores.class1 = bg1$pred1, curve = TRUE)
pr2 <- pr.curve(scores.class0 = fg2$pred2, scores.class1 = bg2$pred2, curve = TRUE)
pr3 <- pr.curve(scores.class0 = fg3$pred3, scores.class1 = bg3$pred3, curve = TRUE)

PRcurves <- ggplot() +
  geom_line(data = data.frame(pr$curve), aes(x = X1, y = X2, color = X3)) + 
  geom_line(data = data.frame(pr1$curve), aes(x = X1, y = X2, color = X3)) + 
  geom_line(data = data.frame(pr2$curve), aes(x = X1, y = X2, color = X3)) + 
  geom_line(data = data.frame(pr3$curve), aes(x = X1, y = X2, color = X3)) + 
  labs(x = "Recall",y = "Precision", title = "Precision Recall Curves (All years)", colour = "Threshold") +
  scale_colour_gradient2(low = "white", mid = "grey", high = "black") +
  geom_text(aes(x = 0.21, y = 0.20, label = paste("AUPRC", round(pr$auc.integral, 4))), size = 4, colour = "black") +
  geom_text(aes(x = 0.245, y = 0.175, label = paste("AUPRC lag one", round(pr1$auc.integral, 4))), size = 4, colour = "black") +
  geom_text(aes(x = 0.245, y = 0.15, label = paste("AUPRC lag two", round(pr2$auc.integral, 4))), size = 4, colour = "black") +
  geom_text(aes(x = 0.248, y = 0.125, label = paste("AUPRC lag three", round(pr3$auc.integral, 4))), size = 4, colour = "black") +
  theme_bw(base_size = 11, base_family = "") +
  geom_hline(yintercept = sum(data_test3$status == 1) / sum(data_test3$status == 0) - 0.01, alpha = 0.3) +
  theme(aspect.ratio = 1)

PRcurves

#n = 4; tiff("PRcurves.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(PRcurves); dev.off()

##################################################################################################################
library(pROC)
roc <- roc(results$testactual, results$pred)
roc1 <- roc(results1$testactual1, results1$pred1)
roc2 <- roc(results2$testactual2, results2$pred2)
roc3 <- roc(results3$testactual3, results3$pred3)

AUC <- round(roc$auc, 4)
AUC1 <- round(roc1$auc, 4)
AUC2 <- round(roc2$auc, 4)
AUC3 <- round(roc3$auc, 4)

roc_curves <- ggroc(list("ROC" = roc, "ROC 1" = roc1, "ROC 2" = roc2, "ROC 3" = roc3), legacy.axes = TRUE, linetype = 1) +
  geom_line(size = 1, alpha = 0.7) +
  scale_colour_discrete(name = "Model") +
  labs(title= "ROC curves (All years)", 
       y = "True Positive Rate (Sensitivity)",
       x = "False Positive Rate (1-Specificity)") +
  geom_abline(show.legend = TRUE, alpha = 0.7) +
  geom_text(aes(x = 0.325, y = 0.20, label = paste("AUC", AUC)), size = 4, colour = "black") +
  geom_text(aes(x = 0.325, y = 0.16, label = paste("   AUC lag one", AUC1)), size = 4, colour = "black") +
  geom_text(aes(x = 0.325, y = 0.12, label = paste("   AUC lag two", AUC2)), size = 4, colour = "black") +
  geom_text(aes(x = 0.325, y = 0.08, label = paste("   AUC lag three", AUC3)), size = 4, colour = "black") +
  theme_bw(base_size = 11, base_family = "")
# theme(aspect.ratio = 1)

roc_curves

#n = 4; tiff("roc_curves.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(roc_curves); dev.off()



###### Last available year
print("Last available year")
conMat$table
print(table(data_test$status))
sum(data_test$status == 1) + sum(data_test$status == 0)
print(paste("Accuracy =", conMat$overall[[1]]))
print(paste("Sensitivity =", conMat$byClass[[1]]))
print(paste("Specificity ", conMat$byClass[[2]]))
print(paste("Precision ", conMat$byClass[[5]]))
print(paste("F1 ", conMat$byClass[[7]]))
print(paste("MCC ", MCC))
print(paste("G Means ", G))
print(paste("Pplus ", Pplus))
print(paste("Pneg ", Pneg))
print(paste("Balanced Accuracy ", BA))
print(paste("Youden Index ", YI))

###### One year from Bankruptcy
print("One year from bankruptcy")
conMat1$table
print(table(data_test1$status))
sum(data_test1$status == 1) + sum(data_test1$status == 0)
print(paste("Accuracy =", conMat1$overall[[1]]))
print(paste("Sensitivity =", conMat1$byClass[[1]]))
print(paste("Specificity ", conMat1$byClass[[2]]))
print(paste("Precision ", conMat1$byClass[[5]]))
print(paste("F1 ", conMat1$byClass[[7]]))
print(paste("MCC ", MCC1))
print(paste("G Means ", G1))
print(paste("Pplus ", Pplus1))
print(paste("Pneg ", Pneg1))
print(paste("Balanced Accuracy ", BA1))
print(paste("Youden Index ", YI1))

###### Two years from bankruptcy
print("Two years from bankruptcy")
conMat2$table
print(table(data_test2$status))
sum(data_test2$status == 1) + sum(data_test2$status == 0)
print(paste("Accuracy =", conMat2$overall[[1]]))
print(paste("Sensitivity =", conMat2$byClass[[1]]))
print(paste("Specificity ", conMat2$byClass[[2]]))
print(paste("Precision ", conMat2$byClass[[5]]))
print(paste("F1 ", conMat2$byClass[[7]]))
print(paste("MCC ", MCC2))
print(paste("G Means ", G2))
print(paste("Pplus ", Pplus2))
print(paste("Pneg ", Pneg2))
print(paste("Balanced Accuracy ", BA2))
print(paste("Youden Index ", YI2))

###### Three years from bankruptcy
print("Three years from bankruptcy")
conMat3$table
print(table(data_test3$status))
sum(data_test3$status == 1) + sum(data_test3$status == 0)
print(paste("Accuracy =", conMat3$overall[[1]]))
print(paste("Sensitivity =", conMat3$byClass[[1]]))
print(paste("Specificity ", conMat3$byClass[[2]]))
print(paste("Precision ", conMat3$byClass[[5]]))
print(paste("F1 ", conMat3$byClass[[7]]))
print(paste("MCC ", MCC3))
print(paste("G Means ", G3))
print(paste("Pplus ", Pplus3))
print(paste("Pneg ", Pneg3))
print(paste("Balanced Accuracy ", BA3))
print(paste("Youden Index ", YI3))


density_plot <- ggplot() + 
  geom_density(data = results %>%
                 filter(testactual == 0), aes(pred), color='blue', fill = 'blue', alpha  = 0.3) + 
  geom_density(data = results %>%
                 filter(testactual == 1), aes(pred), color='red', fill = 'red', alpha = 0.3) +
  labs(x = "Prediction probability",y = "Density", title = "Density plot: Last available year") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1)

#n = 4; tiff("density_plot.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(density_plot); dev.off()


################################################################################################################
#Overall model variable importance
importancexgb_data <- xgb.importance(colnames(x_train), model = model)
importancexgb <- importancexgb_data
importancexgb <- xgb.ggplot.importance(importancexgb, top_n = 10, n_clusters = 1) +
  labs(title = "Variable Importance: Last available year", y = "Importance Gain", x = "Top 10 Variables") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none")

importancexgb

#n = 4; tiff("importancexgb.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(importancexgb); dev.off()


importancexgb1_data <- xgb.importance(colnames(x_train1), model = model1)
importancexgb1 <- importancexgb1_data
importancexgb1 <- xgb.ggplot.importance(importancexgb1, top_n = 10, n_clusters = 1) +
  labs(title = "Variable Importance: Two years before bankruptcy", y = "Importance Gain", x = "Top 10 Variables") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none")

importancexgb1

#n = 4; tiff("importancexgb1.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(importancexgb1); dev.off()


importancexgb2_data <- xgb.importance(colnames(x_train2), model = model2)
importancexgb2 <- importancexgb2_data
importancexgb2 <- xgb.ggplot.importance(importancexgb2, top_n = 10, n_clusters = 1) +
  labs(title = "Variable Importance: Three years before bankruptcy", y = "Importance Gain", x = "Top 10 Variables") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none")

importancexgb2

#n = 4; tiff("importancexgb2.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(importancexgb2); dev.off()


importancexgb3_data <- xgb.importance(colnames(x_train3), model = model3)
importancexgb3 <- importancexgb3_data
importancexgb3 <- xgb.ggplot.importance(importancexgb3, top_n = 10, n_clusters = 1) +
  labs(title = "Variable Importance: Four years before bankruptcy", y = "Importance Gain", x = "Top 10 Variables") +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1) + theme(legend.position="none")

importancexgb3

#n = 4; tiff("importancexgb3.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(importancexgb3); dev.off()

################################################################################################################

importancexgb_ALL <- cbind(importancexgb_data$Feature, importancexgb_data$Gain, 
              importancexgb1_data$Feature, importancexgb1_data$Gain,
              importancexgb2_data$Feature, importancexgb2_data$Gain,
              importancexgb3_data$Feature, importancexgb3_data$Gain)


colnames(importancexgb_ALL) <- c("Feature", "FirstYear", "Feature1", "SecondYear", "Feature2", "ThirdYear", "Feature3", "FourthYear")
importancexgb_ALL <- as.data.frame(importancexgb_ALL)

library(tidyverse)
feature_suffix <- c("", "1", "2", "3")
year_prefix <- c("First", "Second", "Third", "Fourth")

x <- map2(feature_suffix, year_prefix,
          ~ importancexgb_ALL %>% 
            select(feature = paste0("Feature", .x), value = paste0(.y, "Year")) %>%
            mutate(year = paste0(.y, "Year"))
) %>%
  bind_rows(.) %>%
  mutate(value = as.numeric(value))

colnames(x) <- c("Variable", "Gain", "Year")


xy <- x %>% 
  group_by(Year) %>% 
  arrange(Gain) 

xy <- xy %>%
  arrange(match(Year, c("FourthYear", "ThirdYear", "SecondYear", "FirstYear")))

xy$Variable <- sub("2$", "", xy$Variable)
xy$Variable <- sub("3$", "", xy$Variable)
xy$Variable <- sub("4$", "", xy$Variable)


colrs=c()
for (i in unique(xy$Year)){
  gainx=xy$Gain[xy$Year==i]
  colrs=c(colrs,colorRampPalette(c("darkblue", "lightblue" ))(length(gainx))[rank(gainx)])
}


ggplot(xy, aes(x=Year, y=Gain, group=Gain, label=Variable)) +
  geom_bar(stat="identity", color="black", fill=colrs, position="dodge") +
  geom_text(position=position_dodge(width=0.9), hjust=-0.05) +
  ggtitle("Feature Importance (All years)") +
  guides(fill=FALSE) +
  scale_x_discrete(limits=unique(xy$Year)) +
  coord_flip() +
  theme_bw(base_size = 11, base_family = "") +
  scale_fill_gradient2(low="red", mid="yellow", high="green")

##################################### XGBoost Explainer on the Optimised model ##################################
#The XGBoost Explainer
library(xgboostExplainer)

explainer <- buildExplainer(model, dtrain, type = "binary",
                            base_score = 0.5, n_first_tree = model$best_ntreelimit - 1) #Note: nround should be the best_ntreelimit obtained from best_iteration from earlystoppingrounds in xgb.cv

pred.breakdown <- explainPredictions(model, explainer, dtest) # find out what this does, as it is not used in showwaterfall()

cat('Breakdown Complete', '\n')
weights = rowSums(pred.breakdown)
pred.xgb = 1 / (1 + exp(-weights))
cat(max(pred - pred.xgb), '\n')

pred.xgb 

find_bankrupt_firms <- as.data.frame(y_test)
find_bankrupt_firms$prediction <- results$prediction
find_bankrupt_firms$BvD.ID.number <- data_test$BvD.ID.number
find_bankrupt_firms$index <- seq.int(nrow(find_bankrupt_firms))

bankrupt_firms <- find_bankrupt_firms %>%
  filter(y_test == 1) %>%
  filter(prediction == 0)
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
idx_to_get <- as.integer(14415)
y_test[idx_to_get]

greyfirm_1 <- showWaterfall(model, explainer, dtest,
                           data.matrix(x_test), idx_to_get, type = "binary")

greyfirm_1
#n = 4; tiff("Grey_firm_bankrupt_predictedActive.tiff", width=3.5*n, height=2.33*n, units="in", res=3000/n); print(greyfirm_1); dev.off()


##################################################################################################################

x <- as.data.frame(cbind(data_test$DailySALES.EBIT, data_test$status, pred.breakdown$DailySALES.EBIT))

x$Status <- ifelse(x$V2 == 0, "Active", "Bankrupt")


ggplot(x, aes(x = V1, y = V3, colour = Status, shape = Status, alpha = Status)) +
  geom_point(size = 1, na.rm = TRUE) +
  labs(title = "Effect of EBIT / FinExp on the Log-odds performance", y = "Effect on Level Impact on Log-odds", x = "EBIT / FinExp") +
  scale_colour_manual(values = c("deepskyblue1", "firebrick2")) +
  scale_shape_manual(values=c(16, 16)) +
  scale_alpha_manual(values=c(0.15, 0.9)) +
  theme_bw(base_size = 11, base_family = "") +
  theme(aspect.ratio = 1)  # + geom_smooth(method = "loess", span = 0.8, se = TRUE)



