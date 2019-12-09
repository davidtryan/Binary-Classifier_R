# ---
# title: "Binary Classification Modeling"
# author: "David Ryan"
# ---
  
# Make your directory (e.g. “rcourse_lesson3”) with folders inside (e.g. “data”, “figures”, “scripts”, “write_up”).
# Make an R Project based in your main directory folder (e.g. “rcourse_lesson3”).
# Create the repository on Bitbucket 

# Separate into cleaning and figures and modeling scripts?
## READ IN DATA ####
#source("scripts/rcourse_lesson3_cleaning.R")

#clear worksace
rm(list = ls(all = TRUE))

#https://datascienceplus.com/perform-logistic-regression-in-r/
#https://gist.github.com/mick001/ac92e7c017aecff216fd
#https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/
#https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/
# http://amunategui.github.io/binary-outcome-modeling/
# http://amunategui.github.io/binary-outcome-modeling/#sourcecode
# (dplyr) https://datascienceplus.com/r-for-publication-by-page-piccinini-lesson-3-logistic-regression/
# https://www.r-bloggers.com/evaluating-logistic-regression-models/    (look at this for Greman Credit card data)
# https://rstudio-pubs-static.s3.amazonaws.com/43302_2d242dbea93b46c98ed60f6ac8c62edf.html

#Can use categorical and continuous data

# TITANIC DATASET

##############################

## LOAD PACKAGES/LIBRARIES
library (aod)
library (ggplot2)
library (caret)   #autoautomatically fine tune hyper parameters using a grid search
library (pROC)
library (dplyr)

#https://stats.idre.ucla.edu/r/dae/multinomial-logistic-regression/
require(foreign)
require(nnet)   #multinom (multinomial regression - many categories >2) (also mlogit - requires data shaping)
require(ggplot2)
require(reshape2)


#Load
#Examine
#Plot


## LOAD DATA
training.data.raw <- read.csv('train.csv', header=T, na.strings=c(""))   #make sure all missing values are listed as NA
#mydata <- read.csv("http://stats.idre.ucla.edu/stat/data/binary.csv")
# titanicDF <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')
#data(GermanCredit)

#Partition
Train <- createDataPartition(GermanCredit$Class, p=0.6, list=FALSE)
training <- GermanCredit[ Train, ]
testing <- GermanCredit[ -Train, ]


## DATA EXPLORATION
# view the first few rows of the data
head(mydata)
summary(mydata)   #check for continuous, nominal, ordinal, categorical data
print(str(mydata))
sapply(my data, sd)
dim(mydata)

#Check for missing values and look for #unique values 
sapply(training.data.raw,function(x) sum(is.na(x)))
sapply(training.data.raw, function(x) length(unique(x)))  #too many unique (i.e. names) -> get rid of them; a few unique -> categorical (must turn to factors/dummies later)
levels(NH11$hypev) # check levels of hypev
with(ml, do.call(rbind, tapply(write, prog, function(x) c(M = mean(x), SD = sd(x)))))

library(Amelia)
missmap(training.data.raw, main = "Missing values vs observed")

#Empty cells or small cells: You should check for empty or small cells by doing a crosstab between categorical predictors and the outcome variable. If a cell has very few cases (a small cell), the model may become unstable or it might not run at all.
## two-way contingency table of categorical outcome and predictors we want
## to make sure there are not 0 cells
xtabs(~admit + rank, data = mydata)



## DATA CLEANING
#(drop too many missings and too many unique values)
data <- subset(training.data.raw,select=c(2,3,5,6,7,8,10,12))

#Take care of missings
data$Age[is.na(data$Age)] <- mean(data$Age,na.rm=T)  #can replace with avg, median or mode (or use parameter inside fitting function to impute)
#remove rows with missings?
#This is what we will do prior to the stepwise procedure, creating a data frame called Data.omit.  However, when we create our final model, we want to exclude only those observations that have missing values in the variables that are actually included in that final model.  For testing the overall p-value of the final model, plotting the final model, or using the glm.compare function, we will create a data frame called Data.final with only those observations excluded


	 
	 

#make sure categorical variables are factors
is.factor(data$Sex)
is.factor(data$Embarked)
#NH11$hypev <- factor(NH11$hypev, levels=c("2 No", "1 Yes"))
##### FOR CARET - gbm and glmnet ############
######## DUMMIFY FACTOR VARIABLES ###########
#Our data is starting to look good but we have to fix the factor variables as most models only accept numeric data. Again, gbm can deal with factor variables as it will dummify them internally, but glmnet won’t. In a nutshell, dummifying factors breaks all the unique values into separate columns (see my post on Brief Walkthrough Of The dummyVars function from {caret}). This is a caret function
titanicDF$Title <- as.factor(titanicDF$Title)
titanicDummy <- dummyVars("~.",data=titanicDF, fullRank=F)
titanicDF <- as.data.frame(predict(titanicDummy,titanicDF))
print(names(titanicDF))
##############################################

#examine R representation
contrasts(data$Sex)
contrasts(data$Embarked)

#If there are only a few rows with missing vals for a variable, delete those rows
#As for the missing values in Embarked, since there are only two, we will discard those two rows (we could also have replaced the missing values with the mode and keep the datapoints).
data <- data[!is.na(data$Embarked),]
rownames(data) <- NULL

###### Look at outcome proportions
#Look at outcome variable if possible, if <15% events then it is rare event modeling/ (if rare, need to balance train/test sets at 50:50 with outcomes)
prop.table(table(titanicDF$Survived))
#I like generalizing my variables so that I can easily recycle the code for subsequent needs:
outcomeName <- 'Survived'
predictorsNames <- names(titanicDF)[names(titanicDF) != outcomeName]



## DATA PREPARATION
#Split into training and test sets
train <- data[1:800,]
test <- data[801:889,]
# set.seed(88)
# split <- sample.split(train$Recommended, SplitRatio = 0.75)
# dresstrain <- subset(train, split == TRUE)
# dresstest <- subset(train, split == FALSE)

##### SELECTING VARIABLES? ################
# https://www.udemy.com/practical-data-science-reducing-high-dimensional-data-in-r/?couponCode=1111
# http://data.princeton.edu/R/glms.html

#Search for 2-factor interactions?
#search for collinearity / multi-collinearity
#Multiple correlation is one tool for investigating the relationship among potential independent variables.  For example, if two independent variables are correlated to one another, likely both won’t be needed in a final model, but there may be reasons why you would choose one variable over the other.
### Select only those variables that are numeric or can be made numeric

library(dplyr)

Data.num = 
   select(Data,
          Status, 
          Length,
          Mass,
          Range,
          Migr,
          Insect,
          Diet,
          Clutch,
          Broods,
          Wood,
          Upland,
          Water,
          Release,
          Indiv)


### Covert integer variables to numeric variables

Data.num$Status  = as.numeric(Data.num$Status)
Data.num$Length  = as.numeric(Data.num$Length)
Data.num$Migr    = as.numeric(Data.num$Migr)
Data.num$Insect  = as.numeric(Data.num$Insect)
Data.num$Diet    = as.numeric(Data.num$Diet)
Data.num$Broods  = as.numeric(Data.num$Broods)
Data.num$Wood    = as.numeric(Data.num$Wood)
Data.num$Upland  = as.numeric(Data.num$Upland)
Data.num$Water   = as.numeric(Data.num$Water)
Data.num$Release = as.numeric(Data.num$Release)
Data.num$Indiv   = as.numeric(Data.num$Indiv)

#Examine correlations
### Note I used Spearman correlations here

library(PerformanceAnalytics)

chart.Correlation(Data.num, 
                  method="spearman",
                  histogram=TRUE,
                  pch=16)
 
library(psych)

corr.test(Data.num, 
          use = "pairwise",
          method="spearman",
          adjust="none",      # Can adjust p-values; see ?p.adjust for options
          alpha=.05)

#bestglm?

#random forest varImp and boruta package


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

#Automatic methods are useful when the number of explanatory variables is large
#and it is not feasible to fit all possible models. In this case, it is more efficient to
#use a search algorithm (e.g., Forward selection, Backward elimination and
#Stepwise regression) to find the best model.
#http://www.stat.columbia.edu/~martin/W2024/R10.pdf
# a) forward selection
null=lm(Price~1, data=Housing)
full=lm(Price~., data=Housing)
step(null, scope=list(lower=null, upper=full), direction="forward")
# b) Backward selection
step(full, data=Housing, direction="backward")
# c) stepwise regression
step(null, scope = list(upper=full), data=Housing, direction="both")


# compute quantities of interest for predicted values (model) - plots of variables vs. predicted
library(effects)
plot(allEffects(hyp.out))


## MODELING
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

###############################################
###############################################
###### Alternative (caret)
#We’re going to use two models: gbm (Generalized Boosted Models) and glmnet (Generalized Linear Models). Approaching a new data set using different models is one way of getting a handle on your data. Gbm uses boosted trees while glmnet uses regression. 
#changing the outcome variable to a factor (we use a copy of the outcome as we’ll need the original one for our next model):
titanicDF$Survived2 <- ifelse(titanicDF$Survived==1,'yes','nope')
titanicDF$Survived2 <- as.factor(titanicDF$Survived2)
outcomeName <- 'Survived2'
#Set seed
set.seed(1234)
splitIndex <- createDataPartition(titanicDF[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- titanicDF[ splitIndex,]
testDF  <- titanicDF[-splitIndex,]
#Caret offers many tuning functions to help you get as much as possible out of your models; the trainControl function allows you to control the resampling of your data. This will split the training data set internally and do it’s own train/test runs to figure out the best settings for your model. In this case, we're going to cross-validate the data 3 times, therefore training it 3 times on different portions of the data before settling on the best tuning parameters (for gbm it is shrinkage, trees, and interaction depth). You can also set these values yourself if you don’t trust the function.
# create caret trainControl object to control the number of cross-validations performed
objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
#ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)
#mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate + Housing.Own + 
#                   CreditHistory.Critical,  data=GermanCredit, method="glm", family="binomial",
#                 trControl = ctrl, tuneLength = 5)
#pred = predict(mod_fit, newdata=testing)
#confusionMatrix(data=pred, testing$Class)   #http://stackoverflow.com/questions/23806556/caret-train-predicts-very-different-then-predict-glm
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], 
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
################################################
# glmnet model
################################################

# pick model gbm and find out what type of model it is
getModelInfo()$glmnet$type

## save the outcome for the glmnet model
#titanicDF$Survived  <- tempOutcome
outcomeName <- 'Survived'

# split data into training and testing chunks
set.seed(1234)
splitIndex <- createDataPartition(titanicDF[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- titanicDF[ splitIndex,]
testDF  <- titanicDF[-splitIndex,]

# create caret trainControl object to control the number of cross-validations performed
objControl <- trainControl(method='cv', number=3, returnResamp='none')

# run model
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], method='glmnet',  metric = "RMSE", trControl=objControl))

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

# Random Forest
set.seed(2014)

rfModel <- train(Class ~ ., data=trainData, method = "rf", metric="ROC", trControl = fitControl, verbose=FALSE, tuneLength=5)
pred.rfModel <- as.vector(predict(rfModel, newdata=testData, type="prob")[,"yes"])


roc.rfModel <- pROC::roc(testData$Class, pred.rfModel)

auc.rfModel <- pROC::auc(roc.rfModel)
###############################################
###############################################



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


					   
					   
					   
					   
					   
					   
					   
					   
					   
					   
#######################
#########################
###########################
#http://rcompanion.org/rcompanion/e_07.html
# Determining model with step procedure
### Create new data frame with all missing values removed (NA’s)
Data.omit = na.omit(Data)

### Define full and null models and do step procedure
model.null = glm(Status ~ 1, 
                 data=Data.omit,
                 family = binomial(link="logit")
                 )
model.full = glm(Status ~ Length + Mass + Range + Migr + Insect + Diet + 
                          Clutch + Broods + Wood + Upland + Water + 
                          Release + Indiv,
                 data=Data.omit,
                 family = binomial(link="logit")
                 )
step(model.null,
     scope = list(upper=model.full),
             direction="both",
             test="Chisq",
             data=Data)
			 
			 
			 
## Final model (minimized AIC)
model.final = glm(Status ~ Upland + Migr + Mass + Indiv + Insect + Wood,
                  data=Data,
                  family = binomial(link="logit"),
                  na.action(na.omit)
                  )

summary(model.final)
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