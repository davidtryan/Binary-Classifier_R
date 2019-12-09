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
setwd("/Users/dtr5a/Documents/DataScience/LogisticRegressionSetup/")


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


#########################
#########################
## LOAD DATA
training <- read.csv('i360Project/ProjectFiles/trainingF.csv', header=T, stringsAsFactors=FALSE)
training <- training[,-1]   #go back and figure out why training has ID
testing <- read.csv('i360Project/ProjectFiles/testingF.csv', header=T, stringsAsFactors=FALSE)
trainingDum <- read.csv('i360Project/ProjectFiles/trainingF_dum.csv', header=T, stringsAsFactors=FALSE)

# Train <- createDataPartition(training.data.raw$SPENDINGRESPONSE, p=0.6, list=FALSE)
# training <- training.data.raw[Train, ]
# testing <- training.data.raw[-Train, ]


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
trainingF <- trainingF[,names(trainingF)[!names(trainingF)%in%'f2']]
for (i in names(which(sapply(trainingF, function(x) is.integer(x))))) {
  trainingF[,i] <- as.factor(trainingF[,i])
}
# for (i in names(which(sapply(trainingF[,stepwiseVars], function(x) is.character(x))))) {
#   trainingF[,i] <- as.factor(trainingF[,i])
# }
unique(sapply(trainingF, function(x) class(x)))

#Trim categorical variables
catVars <- names(which(sapply(trainingF, function(x) is.factor(x))))
sapply(trainingF[,catVars], function(x) length(unique(x)))
trainingF <- trainingF[,names(trainingF)[!names(trainingF)%in%'f1']]
catVars <- catVars[-which(catVars=='f1')]
sapply(trainingF[,catVars], function(x) length(unique(x)))

train_s <- createDataPartition(trainingF$outcomeName2, p=0.6, list=FALSE)
trainingF_s <- trainingF[train_s, ]
testingF_s <- trainingF[-train_s, ]

training <- training[,c(names(training)[names(training)%in%names(trainingF_s)],'outcomeName')]
testing$outcomeName <- as.numeric(sapply(testing$SPENDINGRESPONSE, function(x) ifelse(x=="Reduce National Debt and Deficit",0,1)))
testing$outcomeName2 <- as.factor(testing$SPENDINGRESPONSE)
testing <- testing[,names(training)[names(training) %in% names(testing)]]
training_sI <- training[train_s, ]
testing_sI <- training[-train_s, ]
training_s <- training_sI[,-which(names(training_sI)=='outcomeName2')]
testing_s <- testing_sI[,-which(names(testing_sI)=='outcomeName2')]


for (i in names(which(sapply(training, function(x) class(x)=='numeric')))) {
  testing[,i] <- as.numeric(testing[,i])
}
for (i in names(which(sapply(training, function(x) class(x)=='factor')))) {
  testing[,i] <- as.factor(testing[,i])
}
names(which(sapply(training, function(x) class(x)=='factor')))

maj <- sample(which(training$outcomeName==0), length(which(training$outcomeName==1)), replace=F)
training_b <- training[c(which(training$outcomeName==1), maj),]


#####################
#####################
## Variable selection
set.seed(1234)
#Fit a glm using a boosting algorithm (as opposed to MLE). Unlike the glm function, glmboost will 
#perform variable selection. After fitting the model, score the test data set and measure the AUC.

null <- glm(outcomeName ~ 1, data=training_s, family=binomial)
full <- glm(outcomeName ~ ., data=training_s, family=binomial)

      # a) forward selection
      fwdsel <- step(null, scope=list(lower=null, upper=full), direction="forward", trace=0)
      summary(fwdsel)
      # b) Backward selection
      backsel <- step(full, data=Housing, direction="backward", trace=0)
      summary(backsel)
# c) stepwise regression
#step(null, scope = list(upper=full), data=Housing, direction="both")
      #stepwise2 <- step(null, scope = list(upper=full), data=training, direction="both", trace=0, test="Chisq")
stepwise <- step(null, scope = list(upper=full), data=training_s, direction="both", trace=0)
summary(stepwise)
varImp(stepwise)

# display variable importance on a +/- scale 
vimp <- varImp(stepwise, scale=F) 
results <- data.frame(row.names(vimp),vimp$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight, decreasing=TRUE),]
results <- results[(results$Weight != 0),]

par(mar=c(5,15,4,2)) # increase y-axis margin. 
xx <- barplot(results$Weight, width = 0.85, 
              main = paste("Variable Importance -",outcomeName), horiz = T, 
              xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
              col = ifelse((results$Weight > 0), 'blue', 'red')) 
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)  

# Select top predictors (>2 varImp)
names(unlist(stepwise[[1]]))
stepwiseVars <- c('f3', 'f95', 'f6', 'f35', 'f73', 'f91', 'f56',
                  'f29', 'f25', 'f42', 'f84', 'f66', 'f76', 'f75')
# stepwiseVars <- c('f95', 'f6', 'f35', 'f73', 'f42', 'f84', 'f56',
#                   'f29', 'f66', 'f76')

##############
##############

library(Boruta)
# Decide if a variable is important or not using Boruta
boruta_output <- Boruta(outcomeName ~ ., data=na.omit(training_s), doTrace=2)  # perform Boruta search
# Confirmed 10 attributes: Humidity, Inversion_base_height, Inversion_temperature, Month, Pressure_gradient and 5 more.
# Rejected 3 attributes: Day_of_month, Day_of_week, Wind_speed.
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed")])  # collect Confirmed and Tentative variables
# boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])  # collect Confirmed and Tentative variables
print(boruta_signif)  # significant variables
#=> [1] "Month"                 "ozone_reading"         "pressure_height"      
#=> [4] "Humidity"              "Temperature_Sandburg"  "Temperature_ElMonte"  
#=> [7] "Inversion_base_height" "Pressure_gradient"     "Inversion_temperature"
#=> [10] "Visibility"
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  # plot variable importance

##############
##############

glmBoruta <- glm(outcomeName ~ ., data=training[,c(boruta_signif,'outcomeName')], family=binomial)
library(car)
vif(glmBoruta)

vif_results <- vif(glmBoruta)[order(vif(glmBoruta)[,3],decreasing=T),]
vif_results[(which(rownames(vif_results)=="f46")+1):nrow(vif_results),]
View(head(training[,rownames(vif_results)]))
sapply(training[,rownames(vif_results)[1:12]], function(x) class(x))
suspect_boruta <- rownames(vif_results)[1:12]
corvals <- data.frame(cor(training[,rownames(vif_results)[1:12]]))
sapply(corvals, function(x) sum(abs(x)))
boruta2 <- rownames(vif_results)
boruta2 <- boruta2[-which(boruta2=='f43')]
(corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta=='f43')]])))
sapply(corvals, function(x) sum(abs(x)))
boruta2 <- boruta2[-which(boruta2=='f34')]
(corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f34','f43'))]])))
sapply(corvals, function(x) sum(abs(x)))
boruta2 <- boruta2[-which(boruta2=='f44')]
(corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f44','f34','f43'))]])))
sapply(corvals, function(x) sum(abs(x)))
boruta2 <- boruta2[-which(boruta2=='f49')]
(corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f49','f44','f34','f43'))]])))
sapply(corvals, function(x) sum(abs(x)))
boruta2 <- boruta2[-which(boruta2=='f35')]
(corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f35','f49','f44','f34','f43'))]])))
sapply(corvals, function(x) sum(abs(x)))
sapply(corvals, function(x) max(abs(x)))
boruta2 <- boruta2[-which(boruta2=='f46')]
(corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f46','f35','f49','f44','f34','f43'))]])))
sapply(corvals, function(x) sum(abs(x)))
sapply(corvals, function(x) max(abs(x)))
boruta2 <- boruta2[-which(boruta2=='f34')]
(corvals <- data.frame(cor(training[,suspect_boruta[-which(suspect_boruta%in%c('f34', 'f46','f35','f49','f44','f34','f43'))]])))
sapply(corvals, function(x) sum(abs(x)))
sapply(corvals, function(x) max(abs(x)))

# boruta2 <- rownames(vif_results)[(nrow(vif_results)-9):nrow(vif_results)]
vif_results[boruta2,]

glmBoruta2 <- glm(outcomeName ~ ., data=training[,c(boruta2,'outcomeName')], family=binomial)
vif(glmBoruta2)
vif_results2 <- vif(glmBoruta2)[order(vif(glmBoruta2)[,3],decreasing=T),]

durbinWatsonTest(glmBoruta2)
crPlots(glmBoruta2)

cor(training_s[,boruta2])

#stepwiseVars <- c('f95', 'f52', 'f42', 'f96', 'f93', 'f80', 'f77', 'f69', 'f81', 'f75')
#stepwiseVars <- c('f95', 'f35', 'f6', 'f101', 'f2', 'f96', 'f73','f77', 'f102', 'f93', 'f21', 'f25')

glmStep <- glm(outcomeName ~ ., data=training[,c(boruta2,'outcomeName')], family=binomial)

#Plot ROC for training and AUC for training
library(ROCR)
p <- predict(glmStep, newdata=training[,c(boruta2,'outcomeName')], type="response")
pr <- prediction(p, training$outcomeName)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf, colorize=TRUE, text.adj=c(-0.2, 1.7))

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

predCat <- ifelse(p>0.40,1,0)
(cm <- confusionMatrix(data=predCat, as.character(training$outcomeName)))


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
## GLM Boost Model
glmBoostModel <- train(outcomeName2 ~ ., data = training_b[,c(boruta2, 'outcomeName2')], 
                       method = "glmboost", metric="ROC", 
                       trControl = fitControl, tuneLength=5, center=TRUE, 
                       family=Binomial(link = c("logit")))
pred.glmBoostModel <- as.vector(predict(glmBoostModel, newdata=training[,c(boruta2, 'outcomeName2')], 
                                        type="prob")[,"Spend"])
(roc.glmBoostModel <- pROC::roc(training$outcomeName2, pred.glmBoostModel))
(auc.glmBoostModel <- pROC::auc(roc.glmBoostModel))

varImp(glmBoostModel)
glmBoostVars <- c('f3', 'f51', 'f73', 'f95', 'f101', 'f92', 'f36', 'f48', 'f37')
intersect(glmBoostVars, stepwiseVars)

glmBoostModel <- train(outcomeName2 ~ ., data = training_b[,c(glmBoostVars, 'outcomeName2')], 
                       method = "glmboost", metric="ROC", 
                       trControl = fitControl, tuneLength=5, center=TRUE, 
                       family=Binomial(link = c("logit")))

pred.glmBoostModel <- as.vector(predict(glmBoostModel, newdata=testingF_s[,c(boruta2, 'outcomeName2')], 
                                        type="prob")[,"Spend"])
predCat <- ifelse(pred.glmBoostModel>0.50,"Spend","Reduce")
(cm <- confusionMatrix(data=predCat, as.character(testingF_s$outcomeName2)))
(roc.glmBoostModel <- pROC::roc(training$outcomeName2, pred.glmBoostModel))
(auc.glmBoostModel <- pROC::auc(roc.glmBoostModel))


#######################
## GLMNET Model
selVars <- stepwiseVars
#selVars <- intersect(stepwiseVars, glmBoostVars)
glmnetModel <- train(outcomeName2 ~ ., data = training_b[,c(glmBoostVars,'outcomeName2')],
                     method='glmnet',  metric = "ROC", trControl=fitControl)
pred.glmnetModel <- as.vector(predict(glmnetModel, newdata=training[,c(boruta2,'outcomeName2')], type="prob")[,"Spend"])
predCat <- ifelse(pred.glmnetModel>0.50,"Spend","Reduce")
(cm <- confusionMatrix(data=predCat, as.character(training$outcomeName2)))
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.glmnetModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, testingF_s$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.glmnetModel <- pROC::roc(testingF_s$outcomeName2, pred.glmnetModel))
(auc.glmnetModel <- pROC::auc(roc.glmnetModel))
plot(roc.glmnetModel)




##########
###########
set.seed(1234)
pc <- prcomp(training[,c(boruta2)])




rfModel <- train(outcomeName2 ~ ., data = training[,c(glmBoostVars,'outcomeName2')],
                 method = "rf", metric="ROC", 
                 trControl = fitControl, verbose=FALSE)
pred.rfModel <- as.vector(predict(rfModel, newdata=testingF_s[,c(glmBoostVars,'outcomeName2')], type="prob")[,"Spend"])
predCat <- ifelse(pred.rfModel>0.025,"Spend","Reduce")
(cm <- confusionMatrix(data=predCat, as.character(testingF_s$outcomeName2)))
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.rfModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, training$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.rfModel <- pROC::roc(testingF_s$outcomeName2, pred.rfModel))
(auc.rfModel <- pROC::auc(roc.rfModel))
plot(roc.rfModel)




#######################
## Random Forest Model
fitControl_s <- trainControl(method = "repeatedcv",
                             number = 2,
                             repeats = 1,   #originally 10
                             ## Estimate class probabilities
                             classProbs = TRUE,
                             ## Evaluate performance using 
                             ## the following function
                             summaryFunction = twoClassSummary)
rfModel_Minit <- train(outcomeName2 ~ ., data = trainingF_s, 
                       method = "rf", metric="ROC", 
                       trControl = fitControl_s, tuneLength=5, verbose=FALSE)
pred.rfModel_Minit <- as.vector(predict(rfModel_Minit, newdata=testingF_s, 
                                        type="prob")[,"Spend"])
(roc.rfModel_Minit <- pROC::roc(testingF_s$outcomeName2, pred.rfModel_Minit))
(auc.rfModel_Minit <- pROC::auc(roc.rfModel_Minit))

varImp(rfModel_Minit)
rfVars <- c('f35', 'f4', 'f36', 'f5', 'f52', 'f15', 'f34', 'f6', 'f16', 'f86')
intersect(stepwiseVars, rfVars)
intersect(glmBoostVars, rfVars)
intersect(intersect(glmBoostVars, stepwiseVars), rfVars)

selVars <- intersect(stepwiseVars, glmBoostVars)

#######################
## GLM Model (with GLM Boost variables)
# Create logreg model with caret and selected variables
glmModel <- train(outcomeName2 ~ ., data = trainingF_s[,c(glmBoostVars,'outcomeName2')], 
                  method="glm", metric="ROC", 
                  trControl = fitControl, tuneLength=5,
                  family="binomial")
# Warning messages:
#   1: In predict.lm(object, newdata, se.fit, scale = 1, type = ifelse(type ==  ... :
#                                                                        prediction from a rank-deficient fit may be misleading
pred.glmModel <- as.vector(predict(glmModel, newdata=testingF_s[,c(glmBoostVars,'outcomeName2')], type="prob")[,"Spend"])
predCat <- ifelse(pred.glmModel>0.5,"Spend","Reduce")
library(e1071)
(cm <- confusionMatrix(data=predCat, testingF_s$outcomeName2))
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.glmModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, trainingF$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.glmModel <- pROC::roc(testingF_s$outcomeName2, pred.glmModel))
(auc.glmModel <- pROC::auc(roc.glmModel))

### Create combined variables list to assess with random forest
selVars <- unique(c(glmBoostVars, stepwiseVars))


















#######################
## Random Forest Model
selVars <- selVars[selVars %in% names(trainingF)]
fitControl_s <- trainControl(method = "repeatedcv",
                           number = 2,
                           repeats = 1,   #originally 10
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

rfModel_Minit <- train(outcomeName2 ~ ., data = trainingF_s, 
                   method = "rf", metric="ROC", 
                   trControl = fitControl_s, verbose=FALSE)
#Model assessment
pred.rfModel <- as.vector(predict(rfModel_Minit, newdata=trainingF_s, type="prob")[,"Spend"])     #or trainingF[-train_s, ]
predCat <- ifelse(pred.rfModel>0.5,"Spend","Reduce")
library(e1071)
(cm <- confusionMatrix(data=predCat, trainingF_s$outcomeName2))
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.rfModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, trainingF_s$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.rfModel <- pROC::roc(trainingF_s$outcomeName2, pred.rfModel))
(auc.rfModel <- pROC::auc(roc.rfModel))
plot(roc.rfModel)

varImp(rfModel_Minit)
vimp <- varImp(rfModel_Minit)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight, decreasing=T),]
results <- results[(results$Weight != 0),]
rfVars <- as.character(results$VariableName[1:10])

## Rerun random forest with only select variables
rfModel_Minit2 <- train(outcomeName2 ~ ., data = trainingF_s[,c(rfVars,'outcomeName2')],
                       method = "rf", metric="ROC", 
                       trControl = fitControl_s, verbose=FALSE)
#Model assessment
pred.rfModel <- as.vector(predict(rfModel_Minit2, newdata=trainingF_s[,c(rfVars,'outcomeName2')], type="prob")[,"Spend"])
predCat <- ifelse(pred.rfModel>0.5,"Spend","Reduce")
library(e1071)
cm <- confusionMatrix(data=predCat, trainingF_s[,c(rfVars,'outcomeName2')]$outcomeName2)
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.rfModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, trainingF_s[,c(rfVars,'outcomeName2')]$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.rfModel <- pROC::roc(trainingF_s[,c(rfVars,'outcomeName2')]$outcomeName2, pred.rfModel))
(auc.rfModel <- pROC::auc(roc.rfModel))
plot(roc.rfModel)
#Model assessment - non-training data
train_test <- trainingF[-train_s, ]
train_test <- train_test[,c(rfVars,'outcomeName2')]
pred.rfModel <- as.vector(predict(rfModel_Minit2, newdata=train_test, type="prob")[,"Spend"])
predCat <- ifelse(pred.rfModel>0.5,"Spend","Reduce")
library(e1071)
cm <- confusionMatrix(data=predCat, train_test$outcomeName2)
for (i in seq(0.1,0.9,0.1)) {
  predCat <- ifelse(pred.rfModel>i,"Spend","Reduce")
  cm <- confusionMatrix(data=predCat, train_test$outcomeName2)
  print(sprintf("Separator of %f yields accuracy of %f", i, as.numeric(cm$overall[1])))
}
(roc.rfModel <- pROC::roc(train_test$outcomeName2, pred.rfModel))
(auc.rfModel <- pROC::auc(roc.rfModel))
plot(roc.rfModel)

    rfModel_Minit3 <- train(outcomeName2 ~ ., data = trainingF[,c(rfVars,'outcomeName2')],
                            method = "rf", metric="ROC", 
                            trControl = fitControl_s, verbose=FALSE)
    
    rfModel <- train(outcomeName2 ~ ., data = trainingF[,c(rfVars,'outcomeName2')],
                            method = "rf", metric="ROC", 
                            trControl = fitControl, verbose=FALSE)






















################################################
# glmnet model
################################################


# get predictions on your testing data
predictions <- predict(object=objModel, testDF[,predictorsNames])

library(pROC)
auc <- roc(testDF[,outcomeName], predictions)
print(auc$auc)
####
library(pROC)
# Compute AUC for predicting Class with the variable CreditHistory.Critical
f1 = roc(Class ~ CreditHistory.Critical, data=training) 
plot(f1, col="red")
###

postResample(pred=predictions, obs=testDF[,outcomeName])

# find out variable importance
summary(objModel)
plot(varImp(objModel,scale=F))

# find out model details
objModel

# display variable importance on a +/- scale 
vimp <- varImp(objModel, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]
results <- results[(results$Weight != 0),]

par(mar=c(5,15,4,2)) # increase y-axis margin. 
xx <- barplot(results$Weight, width = 0.85, 
              main = paste("Variable Importance -",outcomeName), horiz = T, 
              xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
              col = ifelse((results$Weight > 0), 'blue', 'red')) 
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)  


#########
roc.glmModel <- pROC::roc(testData$Class, pred.glmModel)

auc.glmModel <- pROC::auc(roc.glmModel)
###########


################################################
# advanced stuff
################################################

# boosted tree model (gbm) adjust learning rate and and trees
gbmGrid <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = 50,
                        shrinkage = 0.01)

# run model
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], method='gbm', trControl=objControl, tuneGrid = gbmGrid, verbose=F)

# get predictions on your testing data
predictions <- predict(object=objModel, testDF[,predictorsNames])

library(pROC)
auc <- roc(testDF[,outcomeName], predictions)
print(auc$auc)
###############################################
###############################################
###############################################
###############################################
################################################
# glmboost model
################################################
#Fit a glm using a boosting algorithm (as opposed to MLE). Unlike the glm function, glmboost will perform variable selection. After fitting the model, score the test data set and measure the AUC.
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(2014)

glmBoostModel <- train(Class ~ ., data=trainData, method = "glmboost", metric="ROC", trControl = fitControl, tuneLength=5, center=TRUE, family=Binomial(link = c("logit")))

pred.glmBoostModel <- as.vector(predict(glmBoostModel, newdata=testData, type="prob")[,"yes"])

roc.glmBoostModel <- pROC::roc(testData$Class, pred.glmBoostModel)

auc.glmBoostModel <- pROC::auc(roc.glmBoostModel)
###############################################
###############################################
pred.gbmModel <- as.vector(predict(gbmModel, newdata=testData, type="prob")[,"yes"])


roc.gbmModel <- pROC::roc(testData$Class, pred.gbmModel)

auc.gbmModel <- pROC::auc(roc.gbmModel)

###############################################
###############################################

































































# create caret trainControl object to control the number of cross-validations performed
objControl <- trainControl(method='cv', number=3, returnResamp='none', 
                           summaryFunction = twoClassSummary, classProbs = TRUE)
#objControl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)
objModel <- train(training[,stepwiseVars], training[,outcomeName], 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))
#mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate + Housing.Own + 
#                   CreditHistory.Critical,  data=training, method="glm", family="binomial")
#exp(coef(mod_fit$finalModel))   #This informs us that for every one unit increase in Age, the odds of having good credit increases by a factor of 1.01.
##### OR
#mod_fit_one <- glm(Class ~ Age + ForeignWorker + Property.RealEstate + Housing.Own + 
#                     CreditHistory.Critical, data=training, family="binomial")
#mod_fit_two <- glm(Class ~ Age + ForeignWorker, data=training, family="binomial")

#call summary() function on our model to find out what variables were most important:
summary(model)
#tuning parameters were most important to the model (notice the last lines about trees, shrinkage and interaction depth:
print(objModel)
#################################################
# evalutate model
#################################################
# get predictions on your testing data

# class prediction
predictions <- predict(object=objModel, testDF[,predictorsNames], type='raw')
head(predictions)
postResample(pred=predictions, obs=as.factor(testDF[,outcomeName]))

# probabilities 
predictions <- predict(object=objModel, testDF[,predictorsNames], type='prob')
head(predictions)
postResample(pred=predictions[[2]], obs=ifelse(testDF[,outcomeName]=='yes',1,0))
auc <- roc(ifelse(testDF[,outcomeName]=="yes",1,0), predictions[[2]])
print(auc$auc)
###############################################

###############################################
#mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate + Housing.Own + 
#                   CreditHistory.Critical,  data=GermanCredit, method="glm", family="binomial",
#                 trControl = ctrl, tuneLength = 5)
#pred = predict(mod_fit, newdata=testing)
#confusionMatrix(data=pred, testing$Class)   #http://stackoverflow.com/questions/23806556/caret-train-predicts-very-different-then-predict-glm















####### !!!!!






#use RF to select variable importance

#Search for 2-factor interactions?
#search for collinearity / multi-collinearity
#Multiple correlation is one tool for investigating the relationship among potential independent variables.  For example, if two independent variables are correlated to one another, likely both won’t be needed in a final model, but there may be reasons why you would choose one variable over the other.


#bestglm?
#random forest varImp and boruta package


#stepAIC

#no to stepwise 
#http://www.nesug.org/proceedings/nesug07/sa/sa07.pdf
# LASSO for logistic regression

# Stepwise regression - forward vs back?
#It is often advised to not blindly follow a stepwise procedure, but to also compare competing models using fit statistics (AIC, AICc, BIC), or to build a model from available variables that are biologically or scientifically sensible. (http://rcompanion.org/rcompanion/e_07.html)

# FWDselect (https://journal.r-project.org/archive/2016-1/sestelo-villanueva-meiramachado-etal.pdf)
# PCA (prcomp and GBM)
# Random forest - method = 'Boruta' (caret)
# variable selection with GBM and GLMNET - method = 'glmStepAIC' (backward direction is default)
# variable selection with fscaret
# minimum redundancy maximum relevance (mRMRe)

#library(leaps)       #leaps may not work with logistic regression???
#leaps=regsubsets(Price~Size+Lot+Bedrooms+Baths+Taxes, data=Housing, nbest=10)

#library(leaps)
#leaps=regsubsets(Price~Size+Lot+Bedrooms+Baths+Taxes,data=Housing, nbest=10)










# compute quantities of interest for predicted values (model) - plots of variables vs. predicted
library(effects)
plot(allEffects(hyp.out))

#########################
#########################
## MODELING
mod <- glm(SPENDINGRESPONSE ~ ., data=training, family=binomial)


# Optimize coefficients so that hte model gives the best reproduction of training set labels (maximum likelihood or stochastic gradient descent)
#alpha controls location of sigmoid midpoint; beta controls slope of rise (large -> sharp slope up)
#one dimension: Sigmoid curve / two dimensions: decision boundary
#Consider: (from https://stats.idre.ucla.edu/r/dae/logit-regression/)
#1) Logistic regression
#2) SVM
#3) Random Forest
#4) Naive bayes
model <- glm(Survived ~.,family=binomial(link='logit'),data=train)
#mylogit <- glm(admit ~ gre + gpa + rank, data = mydata, family = "binomial")
summary(model)
#The logistic regression coefficients give the change in the log odds of the outcome for a one unit increase in the predictor variable
#EX: For every one unit change in gre, the log odds of admission (versus non-admission) increases by 0.002.
#EX: The indicator variables for rank have a slightly different interpretation. For example, having attended an undergraduate institution with rank of 2, versus an institution with a rank of 1, changes the log odds of admission by -0.675.

fit <- glm(F~x1+x2+x3,data=mydata,family=binomial())
summary(fit) # display results
confint(fit) # 95% CI for the coefficients
exp(coef(fit)) # exponentiated coefficients
exp(confint(fit)) # 95% CI for exponentiated coefficients
predict(fit, type="response") # predicted values
residuals(fit, type="deviance") # residuals

#transform the coefficients to make them easier to interpret
hyp.out.tab <- coef(summary(hyp.out))
hyp.out.tab[, "Estimate"] <- exp(coef(hyp.out))
hyp.out.tab













#########################
#########################
## EVALUATION
#z value is the Wald statistic that tests the hypothesis that the estimate is zero
#http://www.theanalysisfactor.com/r-glm-model-fit/   (Deviance, Fisher scoring, AIC, Homer Lemeshow)
#Look at sigmoid functions with plotted data (for 2 variable pairs)?
# more trails. This can be achieved by using the “expand.grid” function which will be more useful especially with advanced models like random forest, neural networks, support vector machines etc.

#Compare models
test.auc <- data.frame(model=c("glm","glmboost","gbm","glmnet","earth","cart","ctree","rForest"),auc=c(auc.glmModel, auc.glmBoostModel, auc.gbmModel, auc.eNetModel, auc.earthModel, auc.cartModel, auc.partyModel, auc.rfModel))

test.auc <- test.auc[order(test.auc$auc, decreasing=TRUE),]

test.auc$model <- factor(test.auc$model, levels=test.auc$model)

test.auc

library(ggplot2)
theme_set(theme_gray(base_size = 18))
qplot(x=model, y=auc, data=test.auc, geom="bar", stat="identity", position = "dodge")+ geom_bar(fill = "light blue", stat="identity")

plot(rfModel)
plot(gbmModel)   # gbm models are difficult to implement and extremely difficult to interpret.


#https://www.r-bloggers.com/evaluating-logistic-regression-models/
# Likelihood Ratio Test - compare models with different numbers of predictors to make sure model with more predictors does indeed perform better
anova(mod_fit_one, mod_fit_two, test ="Chisq")
library(lmtest)
lrtest(mod_fit_one, mod_fit_two)
# Pseudo R2 (see below)
#While no exact equivalent to the R2 of linear regression exists, the McFadden R2 index can be used to assess the model fit.
library(pscl)
pR2(model)
#Homer-Lemeshow Test
library(MKmisc)
HLgof.test(fit = fitted(mod_fit_one), obs = training$Class)
library(ResourceSelection)
hoslem.test(training$Class, fitted(mod_fit_one), g=10)

#Select models that minimize AIC (http://rcompanion.org/rcompanion/e_07.html)
#1) AIC (Akaike Information Criteria) - The analogous metric of adjusted R2 in logistic regression is AIC. AIC is the measure of fit which penalizes model for the number of model coefficients. Therefore, we always prefer model with minimum AIC value.
#2) Null Deviance and Residual Deviance - Null Deviance indicates the response predicted by a model with nothing but an intercept. Lower the value, better the model. Residual deviance indicates the response predicted by a model on adding independent variables. Lower the value, better the model.
#3) Confusion Matrix - It is nothing but a tabular representation of Actual vs Predicted values. This helps us to find the accuracy of the model and avoid overfitting. This is how it looks like:
#4) ROC Curve - Receiver Operating Characteristic(ROC) summarizes the model's performance by evaluating the trade offs between true positive rate (sensitivity) and false positive rate(1- specificity). For plotting ROC, it is advisable to assume p > 0.5 since we are more concerned about success rate. ROC summarizes the predictive power for all possible values of p > 0.5.  The area under curve (AUC), referred to as index of accuracy(A) or concordance index, is a perfect performance metric for ROC curve. Higher the area under curve, better the prediction power of the model.
# Note: For model performance, you can also consider likelihood function. It is called so, because it selects the coefficient values which maximizes the likelihood of explaining the observed data. It indicates goodness of fit as its value approaches one, and a poor fit of the data as its value approaches zero.

# Predictors with p-vals>0.05 are not significant; lowest p-vals have strongest association to outcome

#Remember that in the logit model the response variable is log odds: ln(odds) = ln(p/(1-p)) = a*x1 + b*x2 + .. + z*xn. Since male is a dummy variable, being male reduces the log odds by 2.75 while a unit increase in age reduces the log odds by 0.037.

#The difference between the null deviance and the residual deviance shows how our model is doing against the null model (a model with only the intercept). The wider this gap, the better. Analyzing the table we can see the drop in deviance when adding each variable one at a time. Again, adding Pclass, Sex and Age significantly reduces the residual deviance. The other variables seem to improve the model less even though SibSp has a low p-value. A large p-value here indicates that the model without the variable explains more or less the same amount of variation. Ultimately what you would like to see is a significant drop in deviance and the AIC.

anova(model, test='Chisq')

#While no exact equivalent to the R2 of linear regression exists, the McFadden R2 index can be used to assess the model fit.
library(pscl)
pR2(model)

#Odds ratio and CI (https://stats.idre.ucla.edu/r/dae/logit-regression/)
## odds ratios and 95% CI
exp(cbind(OR = coef(mylogit), confint(mylogit)))

#Wald statistics for variable importance (https://stats.idre.ucla.edu/r/dae/logit-regression/)
#If the test fails to reject the null hypothesis, this suggests that removing the variable from the model will not substantially harm the fit of that model.
library(survey)
regTermTest(mod_fit_one, "ForeignWorker")
regTermTest(mod_fit_one, "CreditHistory.Critical")

#Variable Importance
varImp(mod_fit)


#found in summary(model):
#This test asks whether the model with predictors fits significantly better than a model with just an intercept (i.e., a null model). The test statistic is the difference between the residual deviance for the model with predictors and the null model. The test statistic is distributed chi-squared with degrees of freedom equal to the differences in degrees of freedom between the current and the null model (i.e., the number of predictor variables in the model). To find the difference in deviance for the two models (i.e., the test statistic) we can use the command:
with(mylogit, null.deviance - deviance)
with(mylogit, df.null - df.residual) #degrees of freedom for the difference between the two models is equal to the number of predictor variables in the mode
with(mylogit, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))
logLik(mylogit) #model's log likelihood





#########################
#########################
## TEST
#By setting the parameter type='response', R will output probabilities in the form of P(y=1|X). Our decision boundary will be 0.5. If P(y=1|X) > 0.5 then y = 1 otherwise y=0. Note that for some applications different decision boundaries could be a better option.
#Here (p/1-p) is the odd ratio. Whenever the log of odd ratio is found to be positive, the probability of success is always more than 50%. A typical logistic model plot is shown below. You can see probability never goes below 0 and above 1.
fitted.results <- predict(model, newdata=subset(test,select=c(2,3,4,5,6,7,8)), type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$Survived)
print(paste('Accuracy',1-misClasificError))

#####
pred = predict(mod_fit, newdata=testing)
accuracy <- table(pred, testing[,"Class"])
sum(diag(accuracy))/sum(accuracy)
## [1] 0.705
pred = predict(mod_fit, newdata=testing)
confusionMatrix(data=pred, testing$Class)
#####

library(caret)
confusionMatrix(data=fitted.results, reference=test$Survived)
# table(fitted.results$Survived, predict > 0.5)
#The 0.84 accuracy on the test set is quite a good result. However, keep in mind that this result is somewhat dependent on the manual split of the data that I made earlier, therefore if you wish for a more precise score, you would be better off running some kind of cross validation such as k-fold cross validation.

#As a last step, we are going to plot the ROC curve and calculate the AUC (area under the curve) which are typical performance measurements for a binary classifier.
#The ROC is a curve generated by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings while the AUC is the area under the ROC curve. As a rule of thumb, a model with good predictive ability should have an AUC closer to 1 (1 is ideal) than to 0.5

library(ROCR)
p <- predict(model, newdata=subset(test,select=c(2,3,4,5,6,7,8)), type="response")
pr <- prediction(p, test$Survived)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf, colorize=TRUE, text.adj=c(-0.2, 1.7))

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#####
library(ROCR)
# Compute AUC for predicting Class with the model
prob <- predict(mod_fit_one, newdata=testing, type="response")
pred <- prediction(prob, testing$Class)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc
####

#plot glm
library(ggplot2)
ggplot(dresstrain, aes(x=Rating, y=Recommended)) + geom_point() + 
  stat_smooth(method="glm", family="binomial", se=FALSE)

#################
#http://tutorials.iq.harvard.edu/R/Rstatistics/Rstatistics.html#orgheadline27
# Create a dataset with predictors set at desired levels
predDat <- with(NH11,
                expand.grid(age_p = c(33, 63),
                            sex = "2 Female",
                            bmi = mean(bmi, na.rm = TRUE),
                            sleep = mean(sleep, na.rm = TRUE)))
# predict hypertension at those levels
cbind(predDat, predict(hyp.out, type = "response",
                       se.fit = TRUE, interval="confidence",
                       newdata = predDat))


					   
					   
					   
					   
					   
					   
					   
					   
					   
					   

#Analysis of variance for individual terms
library(car)
Anova(model.final, type="II", test="Wald")
#Pseudo-R-squared
library(rcompanion)
nagelkerke(model.final)

#Overall p-value for model
### Create data frame with variables in final model and NA’s omitted
library(dplyr)
Data.final = 
   select(Data,
          Status,
          Upland, 
          Migr,
          Mass,
          Indiv,
          Insect,
          Wood)
Data.final = na.omit(Data.final)

### Define null models and compare to final model
model.null = glm(Status ~ 1,
                  data=Data.final,
                  family = binomial(link="logit")
                  )
anova(model.final, 
      model.null, 
      test="Chisq")

library(lmtest)
lrtest(model.final)
 
#Plot of standardized residuals
plot(fitted(model.final), 
     rstandard(model.final))

#Simple plot of predicted values
### Create data frame with variables in final model and NA’s omitted
library(dplyr)
Data.final = 
   select(Data,
          Status,
          Upland, 
          Migr,
          Mass,
          Indiv,
          Insect,
          Wood)

Data.final = na.omit(Data.final)
Data.final$predy = predict(model.final,
                           type="response")

### Plot
plot(Status ~ predy, 
     data = Data.final,
     pch = 16,
     xlab="Predicted probability of 1 response",
     ylab="Actual response")	 

#Check for overdispersion
#Overdispersion is a situation where the residual deviance of the glm is large relative to the residual degrees of freedom.  These values are shown in the summary of the model.  One guideline is that if the ratio of the residual deviance to the residual degrees of freedom exceeds 1.5, then the model is overdispersed.  Overdispersion indicates that the model doesn’t fit the data well:  the explanatory variables may not well describe the dependent variable or the model may not be specified correctly for these data.  If there is overdispersion, one potential solution is to use the quasibinomial family option in glm.
summary(model)
summary(model.final)$deviance / summary(model.final)$df.residual


##Alternative to assess models:  using compare.glm
#An alternative to, or a supplement to, using a stepwise procedure is comparing competing models with fit statistics.  My compare.glm function will display AIC, AICc, BIC, and pseudo-R-squared for glm models.  The models used should all be fit to the same data.  That is, caution should be used if different variables in the data set contain missing values.  If you don’t have any preference on which fit statistic to use, I might recommend AICc, or BIC if you’d rather aim for having fewer terms in the final model. 
#A series of models can be compared with the standard anova function.  Models should be nested within the previous model or the next model in the list in the anova function; and models should be fit to the same data.  When comparing multiple regression models, a p-value to include a new term is often relaxed is 0.10 or 0.15.
#In the following example, the models chosen with the stepwise procedure are used.  Note that while model 9 minimizes AIC and AICc, model 8 minimizes BIC.  The anova results suggest that model 8 is not a significant improvement to model 7.  These results give support for selecting any of model 7, 8, or 9.  Note that the SAS example in the Handbook selected model 4. 
 
### Create data frame with just final terms and no NA’s
library(dplyr)
Data.final = 
   select(Data,
          Status,
          Upland, 
          Migr,
          Mass,
          Indiv,
          Insect,
          Wood)

Data.final = na.omit(Data.final)

### Define models to compare.
model.1=glm(Status ~ 1, 
            data=Data.omit, family=binomial())
model.2=glm(Status ~ Release, 
            data=Data.omit, family=binomial())
model.3=glm(Status ~ Release + Upland, 
            data=Data.omit, family=binomial())
model.4=glm(Status ~ Release + Upland + Migr, 
            data=Data.omit, family=binomial())
model.5=glm(Status ~ Release + Upland + Migr + Mass, 
            data=Data.omit, family=binomial())
model.6=glm(Status ~ Release + Upland + Migr + Mass + Indiv, 
            data=Data.omit, family=binomial())
model.7=glm(Status ~ Release + Upland + Migr + Mass + Indiv + Insect,
            data=Data.omit, family=binomial())
model.8=glm(Status ~ Upland + Migr + Mass + Indiv + Insect, 
            data=Data.omit, family=binomial())
model.9=glm(Status ~ Upland + Migr + Mass + Indiv + Insect + Wood, 
            data=Data.omit, family=binomial())

### Use compare.glm to assess fit statistics.
library(rcompanion)
compareGLM(model.1, model.2, model.3, model.4, model.5, model.6,
           model.7, model.8, model.9)

		   
		   
###########################
##############################

## ENSEMBLING
#http://stackoverflow.com/questions/37647518/combining-binary-classification-algorithms
#https://www.r-bloggers.com/an-intro-to-ensemble-learning-in-r/
#https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/
#https://www.rdocumentation.org/packages/RTextTools/versions/1.4.2/topics/create_analytics