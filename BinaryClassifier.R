# ---
# title: "Binary Classification Modeling"
# author: "David Ryan"
# ---
  
#https://datascienceplus.com/perform-logistic-regression-in-r/
#https://gist.github.com/mick001/ac92e7c017aecff216fd
#https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/
#https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/

# http://amunategui.github.io/binary-outcome-modeling/


#Can use categorical and continuous data

# TITANIC DATASET

##############################

## LOAD PACKAGES/LIBRARIES
library (aod)
library (ggplot2)
library (caret)
library (pROC)


## LOAD DATA
training.data.raw <- read.csv('train.csv', header=T, na.strings=c(""))   #make sure all missing values are listed as NA
#mydata <- read.csv("http://stats.idre.ucla.edu/stat/data/binary.csv")
# titanicDF <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')


## DATA EXPLORATION
# view the first few rows of the data
head(mydata)
summary(mydata)   #check for continuous, nominal, ordinal, categorical data
print(str(mydata))
sapply(my data, sd)

#Check for missing values and look for #unique values 
sapply(training.data.raw,function(x) sum(is.na(x)))
sapply(training.data.raw, function(x) length(unique(x)))

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

#make sure categorical variables are factors
is.factor(data$Sex)
is.factor(data$Embarked)
##### FOR CARET - gbm and glmnet ############
######## DUMMIFY FACTOR VARIABLES ###########
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
prop.table(table(titanicDF$Survived))
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


## MODELING
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


## EVALUATION
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

#plot glm
library(ggplot2)
ggplot(dresstrain, aes(x=Rating, y=Recommended)) + geom_point() + 
  stat_smooth(method="glm", family="binomial", se=FALSE)





