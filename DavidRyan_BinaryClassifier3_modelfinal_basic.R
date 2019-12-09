# ---
# title: "Binary Classification Modeling"
# author: "David Ryan"
# ---

#########################
#########################
## Clear worksace
rm(list = ls(all = TRUE))


#########################
#########################
## Set directory
##setwd("/Users/dtr5a/Documents/DataScience/LogisticRegressionSetup/")


#########################
#########################
## Load libraries of interest
library (aod)
library (ggplot2)
library (caret)   #autoautomatically fine tunes hyper parameters using a grid search
library (pROC)
library (dplyr)
require(foreign)
require(nnet)   #multinom (multinomial regression - many categories >2) (also mlogit - requires data shaping)
require(ggplot2)
require(reshape2)
library(e1071)


#########################
#########################
## LOAD DATA
training <- read.csv('ProjectFiles/training.csv', header=T, stringsAsFactors=FALSE)
trainingDum <- read.csv('i360Project/ProjectFiles/training_dum.csv', header=T, stringsAsFactors=FALSE)


#########################
#########################
## PREPARE DATA FOR MODELING
training$SPENDINGRESPONSE <- as.numeric(as.character(training$SPENDINGRESPONSE))
outcomeName <- 'SPENDINGRESPONSE'
names(training)[which(names(training)==outcomeName)] <- 'outcomeName'
training$randpred <- runif(nrow(training))    #create random variable and select vars that perform better
predictorsNames <- names(training)[names(training) != outcomeName]

class(training$outcomeName)
training$outcomeName2 <- ifelse(training$outcomeName==0,'Reduce', 'Spend')
training$outcomeName2 <- as.factor(training$outcomeName2)
#outcomeName <- 'SPENDINGRESPONSE2'
for (i in names(which(sapply(training, function(x) is.character(x))))) {
  training[,i] <- as.factor(training[,i])
}
trainingF <- training[,!names(training) %in% 'outcomeName']
for (i in names(which(sapply(trainingF, function(x) is.character(x))))) {
  trainingF[,i] <- as.factor(trainingF[,i])
}
sapply(trainingF[,names(which(sapply(trainingF, function(x) is.integer(x))))], function(x) length(unique(x)))
# trainingF <- trainingF[,names(trainingF)[!names(trainingF)%in%'f2']]
for (i in names(which(sapply(trainingF, function(x) is.integer(x))))) {
  trainingF[,i] <- as.factor(trainingF[,i])
}
# for (i in names(which(sapply(trainingF[,stepwiseVars], function(x) is.character(x))))) {
#   trainingF[,i] <- as.factor(trainingF[,i])
# }
unique(sapply(trainingF, function(x) class(x)))

#Trim categorical variables
# catVars <- names(which(sapply(trainingF, function(x) is.factor(x))))
# sapply(trainingF[,catVars], function(x) length(unique(x)))
# trainingF <- trainingF[,names(trainingF)[!names(trainingF)%in%'f1']]
# catVars <- catVars[-which(catVars=='f1')]
# sapply(trainingF[,catVars], function(x) length(unique(x)))

# training <- training[,c(names(training)[names(training)%in%names(trainingF_s)],'outcomeName')]
# testing$outcomeName <- as.numeric(sapply(testing$SPENDINGRESPONSE, function(x) ifelse(x=="Reduce National Debt and Deficit",0,1)))
# testing$outcomeName2 <- as.factor(testing$SPENDINGRESPONSE)
# testing <- testing[,names(training)[names(training) %in% names(testing)]]
# training_sI <- training[train_s, ]
# testing_sI <- training[-train_s, ]
# training_s <- training_sI[,-which(names(training_sI)=='outcomeName2')]
# testing_s <- testing_sI[,-which(names(testing_sI)=='outcomeName2')]

# for (i in names(which(sapply(training, function(x) class(x)=='numeric')))) {
#   testing[,i] <- as.numeric(testing[,i])
# }
# for (i in names(which(sapply(training, function(x) class(x)=='factor')))) {
#   testing[,i] <- as.factor(testing[,i])
# }
# names(which(sapply(training, function(x) class(x)=='factor')))

trainingM <- training
Train <- createDataPartition(trainingM$outcomeName, p=0.8, list=FALSE)
training <- trainingM[Train, ]
testing <- trainingM[-Train, ]

# Create balanced Training sample  
maj <- sample(which(training$outcomeName==0), length(which(training$outcomeName==1)), replace=F)
training_b <- training[c(which(training$outcomeName==1), maj),]
training_bF <- training_b[,-which(names(training_b)=='outcomeName')]
training_b <- training_b[,-which(names(training_b)=='outcomeName2')]
  
  
#####################
#####################
## Variable selection

#1) Stepwise Regression

set.seed(1234)
#Fit a glm using a boosting algorithm (as opposed to MLE). Unlike the glm function, glmboost will 
#perform variable selection. After fitting the model, score the test data set and measure the AUC.

null <- glm(outcomeName ~ 1, data=training_b, family=binomial)
full <- glm(outcomeName ~ ., data=training_b, family=binomial)

      # # a) forward selection
      # fwdsel <- step(null, scope=list(lower=null, upper=full), direction="forward", trace=0)
      # summary(fwdsel)
      # # b) Backward selection
      # backsel <- step(full, data=Housing, direction="backward", trace=0)
      # summary(backsel)

# c) stepwise regression
#step(null, scope = list(upper=full), data=Housing, direction="both")
      #stepwise2 <- step(null, scope = list(upper=full), data=training, direction="both", trace=0, test="Chisq")
stepwise <- step(null, scope = list(upper=full), data=training_b, direction="both", trace=0)
summary(stepwise)
varImp(stepwise)

# display variable importance on a +/- scale 
vimp <- varImp(stepwise, scale=F) 
results <- data.frame(row.names(vimp),vimp$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight, decreasing=TRUE),]
results <- results[(results$Weight != 0),]


# par(mar=c(5,15,4,2)) # increase y-axis margin. 
xx <- barplot(results$Weight, width = 0.85, 
              main = paste("Variable Importance -",outcomeName), horiz = T, 
              xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
              col = ifelse((results$Weight > 0), 'blue', 'red')) 
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)  

# Select top predictors (>2 varImp)
names(unlist(stepwise[[1]]))
stepwiseVars <- c('f3', 'f95', 'f6', 'f32', 'f97', 'f27', 'f19',
                  'f31', 'f69', 'f73', 'f101', 'f94', 'f84', 'f66',
                  'f36', 'f65', 'f67', 'f23', 'f48', 'f102', 'f2')
# 5/24/17
# stepwiseVars <- c('f3', 'f95', 'f5', 'f35', 'f97', 'f92', 'f69',
#                   'f101', 'f98', 'f77', 'f62', 'f25', 'f2', 'f100',
#                   'f66', 'f73', 'f48')
# stepwiseVars2 <- c('f5', 'f95', 'f35', 'f98', 'f100', 'f69', 'f2', 'f3', 'f101', 'f97')
stepwiseVars2 <- c('f36', 'f95', 'f6', 'f97', 'f32', 'f27', 'f101', 'f102', 'f31', 'f19')
intersect(stepwiseVars, stepwiseVars2)
  # stepwiseVars <- c('f95', 'f6', 'f35', 'f73', 'f42', 'f84', 'f56',
#                   'f29', 'f66', 'f76')

##############

# Use RF boruta package to determine variable importance
library(Boruta)
# Decide if a variable is important or not using Boruta
boruta_output <- Boruta(outcomeName ~ ., data=na.omit(training_b), doTrace=2)  # perform Boruta search
# Confirmed 10 attributes: Humidity, Inversion_base_height, Inversion_temperature, Month, Pressure_gradient and 5 more.
# Rejected 3 attributes: Day_of_month, Day_of_week, Wind_speed.
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed")])  # collect Confirmed and Tentative variables
# boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])  # collect Confirmed and Tentative variables
print(boruta_signif)  # significant variables
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  # plot variable importance

(boruta.df <- attStats(boruta_output))
boruta.df <- boruta.df[boruta_signif,]
(boruta.df <- boruta.df[order(boruta.df$meanImp, decreasing=TRUE),])


##############
# Check for multicollinearity and eliminate problem variables
glmBoruta <- glm(outcomeName ~ ., data=training_b[,c(boruta_signif,'outcomeName')], family=binomial)
library(car)
vif(glmBoruta)

vif_results <- vif(glmBoruta)[order(vif(glmBoruta)[,3],decreasing=T),]
# vif_results[(which(rownames(vif_results)=="f46")+1):nrow(vif_results),]
# View(head(training[,rownames(vif_results)]))
sapply(training[,rownames(vif_results)[1:3]], function(x) class(x))
suspect_boruta <- rownames(vif_results)[1:3]
(corvals <- data.frame(cor(training[,rownames(vif_results)[1:3]])))
sapply(corvals, function(x) sum(abs(x)))
boruta2 <- rownames(vif_results)
boruta2 <- boruta2[-which(boruta2=='f34')]
(corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta=='f34')]])))
sapply(corvals, function(x) sum(abs(x)))
# boruta2 <- boruta2[-which(boruta2=='f34')]
# (corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f34','f43'))]])))
# sapply(corvals, function(x) sum(abs(x)))
# boruta2 <- boruta2[-which(boruta2=='f44')]
# (corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f44','f34','f43'))]])))
# sapply(corvals, function(x) sum(abs(x)))
# boruta2 <- boruta2[-which(boruta2=='f49')]
# (corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f49','f44','f34','f43'))]])))
# sapply(corvals, function(x) sum(abs(x)))
# boruta2 <- boruta2[-which(boruta2=='f35')]
# (corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f35','f49','f44','f34','f43'))]])))
# sapply(corvals, function(x) sum(abs(x)))
# sapply(corvals, function(x) max(abs(x)))
# boruta2 <- boruta2[-which(boruta2=='f46')]
# (corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f46','f35','f49','f44','f34','f43'))]])))
# sapply(corvals, function(x) sum(abs(x)))
# sapply(corvals, function(x) max(abs(x)))
# boruta2 <- boruta2[-which(boruta2=='f34')]
# (corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f34', 'f46','f35','f49','f44','f34','f43'))]])))
# sapply(corvals, function(x) sum(abs(x)))
# sapply(corvals, function(x) max(abs(x)))

# boruta2 <- rownames(vif_results)[(nrow(vif_results)-9):nrow(vif_results)]
vif_results[boruta2,]

glmBoruta2 <- glm(outcomeName ~ ., data=training_b[,c(boruta2,'outcomeName')], family=binomial)
vif(glmBoruta2)
(vif_results2 <- vif(glmBoruta2)[order(vif(glmBoruta2)[,3],decreasing=T),])
boruta_ord <- rownames(boruta.df)
boruta_ord <- boruta_ord[-which(boruta_ord=='f34')]
(boruta_ord <- boruta_ord[1:10])
intersect(boruta_ord, boruta2)

glmBoruta_ord <- glm(outcomeName ~ ., data=training_b[,c(boruta_ord,'outcomeName')], family=binomial)
vif(glmBoruta_ord)
(vif_results_ord <- vif(glmBoruta_ord)[order(vif(glmBoruta_ord)[,3],decreasing=T),])

boruta_ord <- boruta_ord[-which(boruta_ord=='f5')]
glmBoruta_ord2 <- glm(outcomeName ~ ., data=training_b[,c(boruta_ord,'outcomeName')], family=binomial)
vif(glmBoruta_ord2)
(vif_results_ord2 <- vif(glmBoruta_ord2)[order(vif(glmBoruta_ord2)[,3],decreasing=T),])

boruta_ord <- boruta_ord[-which(boruta_ord=='f35')]
glmBoruta_ord2 <- glm(outcomeName ~ ., data=training_b[,c(boruta_ord,'outcomeName')], family=binomial)
vif(glmBoruta_ord2)
(vif_results_ord2 <- vif(glmBoruta_ord2)[order(vif(glmBoruta_ord2)[,3],decreasing=T),])

boruta_ord <- boruta_ord[-which(boruta_ord=='f6')]
glmBoruta_ord2 <- glm(outcomeName ~ ., data=training_b[,c(boruta_ord,'outcomeName')], family=binomial)
vif(glmBoruta_ord2)
(vif_results_ord2 <- vif(glmBoruta_ord2)[order(vif(glmBoruta_ord2)[,3],decreasing=T),])


# durbinWatsonTest(glmBoruta2)
# crPlots(glmBoruta2)
# 
# cor(training_s[,boruta2])

#stepwiseVars <- c('f95', 'f52', 'f42', 'f96', 'f93', 'f80', 'f77', 'f69', 'f81', 'f75')
#stepwiseVars <- c('f95', 'f35', 'f6', 'f101', 'f2', 'f96', 'f73','f77', 'f102', 'f93', 'f21', 'f25')

selVars <- boruta_ord


#####################
#####################
## Modeling

glmF <- glm(outcomeName ~ ., data=training_b[,c(selVars,'outcomeName')], family=binomial)

predCat <- ifelse(glmF$fitted.values>0.50,1,0)
(cm <- confusionMatrix(predCat, training_b$outcomeName))

#Plot ROC for training and AUC for training
library(ROCR)
p <- predict(glmF, newdata=testing[,c(selVars,'outcomeName')], type="response")
pr <- prediction(p, testing$outcomeName)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf, colorize=TRUE, text.adj=c(-0.2, 1.7))

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

predCat <- ifelse(p>0.50,1,0)
(cm <- confusionMatrix(data=predCat, as.character(testing$outcomeName)))


###############################
## Setup Caret Models
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 10,   #originally 10
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

#######################
## GLM Model (with GLM Boost variables)
# Create logreg model with caret and selected variables
glmModel <- train(outcomeName2 ~ ., data = training_bF[,c(selVars,'outcomeName2')], 
                  method="glm", metric="ROC", 
                  trControl = fitControl, tuneLength=5,
                  family="binomial")
# Warning messages:
#   1: In predict.lm(object, newdata, se.fit, scale = 1, type = ifelse(type ==  ... :
#                                                                        prediction from a rank-deficient fit may be misleading
pred.glmModel <- as.vector(predict(glmModel, newdata=testing[,c(selVars,'outcomeName2')], type="prob")[,"Spend"])
predCat <- ifelse(pred.glmModel>0.5,"Spend","Reduce")
# library(e1071)
(cm <- confusionMatrix(data=predCat, testing$outcomeName2))
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.glmModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, trainingF$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.glmModel <- pROC::roc(testing$outcomeName2, pred.glmModel))
(auc.glmModel <- pROC::auc(roc.glmModel))

# ### Create combined variables list to assess with random forest
# selVars <- unique(c(glmBoostVars, stepwiseVars))


#######################
## GLM Boost Model
glmBoostModel <- train(outcomeName2 ~ ., data = training_bF[,c(selVars, 'outcomeName2')], 
                       method = "glmboost", metric="ROC", 
                       trControl = fitControl, tuneLength=5, center=TRUE, 
                       family=Binomial(link = c("logit")))
pred.glmBoostModel <- as.vector(predict(glmBoostModel, newdata=testing[,c(selVars, 'outcomeName2')], 
                                        type="prob")[,"Spend"])
(roc.glmBoostModel <- pROC::roc(testing$outcomeName2, pred.glmBoostModel))
(auc.glmBoostModel <- pROC::auc(roc.glmBoostModel))

varImp(glmBoostModel)
# glmBoostVars <- c('f3', 'f51', 'f73', 'f95', 'f101', 'f92', 'f36', 'f48', 'f37')
# intersect(glmBoostVars, stepwiseVars)

# glmBoostModel <- train(outcomeName2 ~ ., data = training_b[,c(glmBoostVars, 'outcomeName2')], 
#                        method = "glmboost", metric="ROC", 
#                        trControl = fitControl, tuneLength=5, center=TRUE, 
#                        family=Binomial(link = c("logit")))
# 
# pred.glmBoostModel <- as.vector(predict(glmBoostModel, newdata=testingF_s[,c(boruta2, 'outcomeName2')], 
#                                         type="prob")[,"Spend"])
predCat <- ifelse(pred.glmBoostModel>0.50,"Spend","Reduce")
(cm <- confusionMatrix(data=predCat, as.character(testing$outcomeName2)))


#######################
## GLMNET Model
# selVars <- stepwiseVars
#selVars <- intersect(stepwiseVars, glmBoostVars)
glmnetModel <- train(outcomeName2 ~ ., data = training_bF[,c(selVars,'outcomeName2')],
                     method='glmnet',  metric = "ROC", trControl=fitControl)
pred.glmnetModel <- as.vector(predict(glmnetModel, newdata=testing[,c(selVars,'outcomeName2')], type="prob")[,"Spend"])
predCat <- ifelse(pred.glmnetModel>0.50,"Spend","Reduce")
(cm <- confusionMatrix(data=predCat, as.character(testing$outcomeName2)))
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.glmnetModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, testingF$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.glmnetModel <- pROC::roc(testing$outcomeName2, pred.glmnetModel))
(auc.glmnetModel <- pROC::auc(roc.glmnetModel))
plot(roc.glmnetModel)


#######################
## GBM Model
# selVars <- stepwiseVars
#selVars <- intersect(stepwiseVars, glmBoostVars)
gbmModel <- train(outcomeName2 ~ ., data = training_bF[,c(selVars,'outcomeName2')],
                     method='gbm',  metric = "ROC", trControl=fitControl)
pred.gbmModel <- as.vector(predict(gbmModel, newdata=testing[,c(selVars,'outcomeName2')], type="prob")[,"Spend"])
predCat <- ifelse(pred.gbmModel>0.50,"Spend","Reduce")
(cm <- confusionMatrix(data=predCat, as.character(testing$outcomeName2)))
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.gbmModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, testingF$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.gbmModel <- pROC::roc(testing$outcomeName2, pred.gbmModel))
(auc.gbmModel <- pROC::auc(roc.gbmModel))
plot(roc.gbmModel)


#######################
## Random Forest Model

rfModel <- train(outcomeName2 ~ ., data = training_bF[,c(selVars,'outcomeName2')],
                 method = "rf", metric="ROC", 
                 trControl = fitControl, verbose=FALSE)
pred.rfModel <- as.vector(predict(rfModel, newdata=testing[,c(selVars,'outcomeName2')], type="prob")[,"Spend"])
predCat <- ifelse(pred.rfModel>0.05,"Spend","Reduce")
(cm <- confusionMatrix(data=predCat, as.character(testing$outcomeName2)))
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.rfModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, training$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.rfModel <- pROC::roc(testing$outcomeName2, pred.rfModel))
(auc.rfModel <- pROC::auc(roc.rfModel))
plot(roc.rfModel)


# 
# 
# #######################
# ## Random Forest Model
# fitControl_s <- trainControl(method = "repeatedcv",
#                              number = 2,
#                              repeats = 1,   #originally 10
#                              ## Estimate class probabilities
#                              classProbs = TRUE,
#                              ## Evaluate performance using 
#                              ## the following function
#                              summaryFunction = twoClassSummary)
# rfModel_Minit <- train(outcomeName2 ~ ., data = trainingF_s, 
#                        method = "rf", metric="ROC", 
#                        trControl = fitControl_s, tuneLength=5, verbose=FALSE)
# pred.rfModel_Minit <- as.vector(predict(rfModel_Minit, newdata=testingF_s, 
#                                         type="prob")[,"Spend"])
# (roc.rfModel_Minit <- pROC::roc(testingF_s$outcomeName2, pred.rfModel_Minit))
# (auc.rfModel_Minit <- pROC::auc(roc.rfModel_Minit))
# 
# varImp(rfModel_Minit)
# rfVars <- c('f35', 'f4', 'f36', 'f5', 'f52', 'f15', 'f34', 'f6', 'f16', 'f86')
# intersect(stepwiseVars, rfVars)
# intersect(glmBoostVars, rfVars)
# intersect(intersect(glmBoostVars, stepwiseVars), rfVars)
# 
# selVars <- intersect(stepwiseVars, glmBoostVars)


# 
# #######################
# ## Random Forest Model
# selVars <- selVars[selVars %in% names(trainingF)]
# fitControl_s <- trainControl(method = "repeatedcv",
#                            number = 2,
#                            repeats = 1,   #originally 10
#                            ## Estimate class probabilities
#                            classProbs = TRUE,
#                            ## Evaluate performance using 
#                            ## the following function
#                            summaryFunction = twoClassSummary)
# 
# rfModel_Minit <- train(outcomeName2 ~ ., data = trainingF_s, 
#                    method = "rf", metric="ROC", 
#                    trControl = fitControl_s, verbose=FALSE)
# #Model assessment
# pred.rfModel <- as.vector(predict(rfModel_Minit, newdata=trainingF_s, type="prob")[,"Spend"])     #or trainingF[-train_s, ]
# predCat <- ifelse(pred.rfModel>0.5,"Spend","Reduce")
# library(e1071)
# (cm <- confusionMatrix(data=predCat, trainingF_s$outcomeName2))
# for (i in seq(0.1,0.9,0.1)) {
#   predCat <- ifelse(pred.rfModel>i,"Spend","Reduce")
#   cm <- confusionMatrix(data=predCat, trainingF_s$outcomeName2)
#   print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
# }
# (roc.rfModel <- pROC::roc(trainingF_s$outcomeName2, pred.rfModel))
# (auc.rfModel <- pROC::auc(roc.rfModel))
# plot(roc.rfModel)
# 
# varImp(rfModel_Minit)
# vimp <- varImp(rfModel_Minit)
# results <- data.frame(row.names(vimp$importance),vimp$importance$Overall)
# results$VariableName <- rownames(vimp)
# colnames(results) <- c('VariableName','Weight')
# results <- results[order(results$Weight, decreasing=T),]
# results <- results[(results$Weight != 0),]
# rfVars <- as.character(results$VariableName[1:10])
# 
# ## Rerun random forest with only select variables
# rfModel_Minit2 <- train(outcomeName2 ~ ., data = trainingF_s[,c(rfVars,'outcomeName2')],
#                        method = "rf", metric="ROC", 
#                        trControl = fitControl_s, verbose=FALSE)
# #Model assessment
# pred.rfModel <- as.vector(predict(rfModel_Minit2, newdata=trainingF_s[,c(rfVars,'outcomeName2')], type="prob")[,"Spend"])
# predCat <- ifelse(pred.rfModel>0.5,"Spend","Reduce")
# library(e1071)
# cm <- confusionMatrix(data=predCat, trainingF_s[,c(rfVars,'outcomeName2')]$outcomeName2)
# for (i in seq(0.1,0.9,0.1)) {
#   predCat <- ifelse(pred.rfModel>i,"Spend","Reduce")
#   cm <- confusionMatrix(data=predCat, trainingF_s[,c(rfVars,'outcomeName2')]$outcomeName2)
#   print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
# }
# (roc.rfModel <- pROC::roc(trainingF_s[,c(rfVars,'outcomeName2')]$outcomeName2, pred.rfModel))
# (auc.rfModel <- pROC::auc(roc.rfModel))
# plot(roc.rfModel)
# #Model assessment - non-training data
# train_test <- trainingF[-train_s, ]
# train_test <- train_test[,c(rfVars,'outcomeName2')]
# pred.rfModel <- as.vector(predict(rfModel_Minit2, newdata=train_test, type="prob")[,"Spend"])
# predCat <- ifelse(pred.rfModel>0.5,"Spend","Reduce")
# library(e1071)
# cm <- confusionMatrix(data=predCat, train_test$outcomeName2)
# for (i in seq(0.1,0.9,0.1)) {
#   predCat <- ifelse(pred.rfModel>i,"Spend","Reduce")
#   cm <- confusionMatrix(data=predCat, train_test$outcomeName2)
#   print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
# }
# (roc.rfModel <- pROC::roc(train_test$outcomeName2, pred.rfModel))
# (auc.rfModel <- pROC::auc(roc.rfModel))
# plot(roc.rfModel)
# 
#     rfModel_Minit3 <- train(outcomeName2 ~ ., data = trainingF[,c(rfVars,'outcomeName2')],
#                             method = "rf", metric="ROC", 
#                             trControl = fitControl_s, verbose=FALSE)
#     
#     rfModel <- train(outcomeName2 ~ ., data = trainingF[,c(rfVars,'outcomeName2')],
#                             method = "rf", metric="ROC", 
#                             trControl = fitControl, verbose=FALSE)



#########################
#########################
## Predict New Values
#z value is the Wald statistic that tests the hypothesis that the estimate is zero
#http://www.theanalysisfactor.com/r-glm-model-fit/   (Deviance, Fisher scoring, AIC, Homer Lemeshow)
#Look at sigmoid functions with plotted data (for 2 variable pairs)?
# more trails. This can be achieved by using the “expand.grid” function which will be more useful especially with advanced models like random forest, neural networks, support vector machines etc.

#Compare models
# test.auc <- data.frame(model=c("glm","glmboost","gbm","glmnet","earth","cart","ctree","rForest"),auc=c(auc.glmModel, auc.glmBoostModel, auc.gbmModel, auc.eNetModel, auc.earthModel, auc.cartModel, auc.partyModel, auc.rfModel))
test.auc <- data.frame(model=c("glm","glmboost","gbm","glmnet"),
                       auc=c(auc.glmModel, auc.glmBoostModel, auc.gbmModel, auc.glmnetModel))
test.auc <- test.auc[order(test.auc$auc, decreasing=TRUE),]
test.auc$model <- factor(test.auc$model, levels=test.auc$model)
test.auc

###Select glmboost model
## GLM Boost Model

## Load and format new data for predicting
pred.raw <- read.csv('ProjectFiles/File3.csv', header=T, na.strings=c(""), stringsAsFactors=FALSE)   #make sure all missing values are listed as NA
predDF <- pred.raw[,c(selVars)]
sapply(predDF, function(x) class(x))
for (i in names(which(sapply(training[,selVars], function(x) class(x))=='numeric'))) {
  predDF[,i] <- as.numeric(predDF[,i])
}
for (i in names(which(sapply(training[,selVars], function(x) class(x))=='factor'))) {
  predDF[,i] <- as.factor(predDF[,i])
}
sapply(predDF, function(x) which(is.na(x)))
predDF$f4[which(is.na(predDF$f4))] <- mean(predDF$f4, na.rm=T)
predDF$f52[which(is.na(predDF$f52))] <- mean(predDF$f52, na.rm=T)
predDF$f36[which(is.na(predDF$f36))] <- mean(predDF$f36, na.rm=T)
sapply(predDF, function(x) which(is.na(x)))


#######################
## GLM Boost Model
glmbVars <- selVars[-which(selVars=='f3')]
glmBoostModel_T <- train(outcomeName2 ~ ., data = training_bF[,c(glmbVars, 'outcomeName2')], 
                       method = "glmboost", metric="ROC", 
                       trControl = fitControl, tuneLength=5, center=TRUE, 
                       family=Binomial(link = c("logit")))
pred.glmBoostModel_T <- as.vector(predict(glmBoostModel_T, newdata=testing[,c(glmbVars, 'outcomeName2')], 
                                        type="prob")[,"Spend"])
predCat <- ifelse(pred.glmBoostModel_T>0.50,"Spend","Reduce")
(cm <- confusionMatrix(data=predCat, as.character(testing$outcomeName2)))
(roc.glmBoostModel_T <- pROC::roc(testing$outcomeName2, pred.glmBoostModel_T))
(auc.glmBoostModel_T <- pROC::auc(roc.glmBoostModel_T))



# Predict, merge and save
finalPred <- as.vector(predict(glmBoostModel_T, newdata=predDF, type="prob")[,"Spend"])
predCat <- ifelse(finalPred>0.5,"Spend to Improve Economy","Reduce National Debt and Deficit")
pred.raw$SPENDINGRESPONSE <- predCat
pred.raw$SPENDINGRESPONSEprob <- finalPred

write.csv(pred.raw, 'ProjectFiles/File3_predictions.csv')
#https://www.rdocumentation.org/packages/RTextTools/versions/1.4.2/topics/create_analytics