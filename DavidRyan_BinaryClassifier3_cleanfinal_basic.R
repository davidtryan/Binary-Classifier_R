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
library(Hmisc)


#########################
#########################
## LOAD DATA
training.data.raw <- read.csv('ProjectFiles/File2.csv', header=T, na.strings=c(""), stringsAsFactors=FALSE)   #make sure all missing values are listed as NA
training.data.raw_response <- read.csv('ProjectFiles/File1.csv', header=T, na.strings=c(""), stringsAsFactors=FALSE)   #make sure all missing values are listed as NA
# Merge features and response data
training.data.raw <- merge(training.data.raw, training.data.raw_response, by=c("ID", "State")) 

# Make appropriate columns numeric
for (i in 1:ncol(training.data.raw)) {
  training.data.raw[,i][which(training.data.raw[,i]=="NULL")]=""
}
for (i in 1:ncol(training.data.raw)) {
  if (all.is.numeric(training.data.raw[,i], what = c("test", "vector"), extras=c('.','NA'))) {
    training.data.raw[,i] = as.numeric(training.data.raw[,i])
  }
}

# Partition
# Divide data into training and testing/validation sets
# Train <- createDataPartition(training.data.raw$SPENDINGRESPONSE, p=0.6, list=FALSE)
# training <- training.data.raw[Train, ]
# testing <- training.data.raw[-Train, ]
training <- training.data.raw


#########################
#########################
## DATA EXPLORATION

# Identify a) binary; b) numeric; c) categorical variables
unique(sapply(training, function(x) class(x)))
numVars <- names(which(sapply(training, function(x) is.numeric(x))))
charVars <- names(which(sapply(training, function(x) is.character(x))))
# a) potential binary variables
names(which(sapply(training[,(numVars)], function(x) length(unique(x)))<4))
unique(training[,names(which(sapply(training[,(numVars)], function(x) length(unique(x)))<4))])   
#mostly NAs - likely no other binary variables, but maybe some categorical that are currently numeric - revist after missings

# View data, its summary, and its properties
dim(training)
str(training)
head(training)
summary(training)   #check for continuous, nominal, ordinal, categorical data
sapply(training[,numVars[-1]], sd, na.rm=T)  #only on numeric values that are not the ID
#with(training, do.call(rbind, tapply(write, prog, function(x) c(M = mean(x), SD = sd(x)))))

# Check for missing values (NAs, blanks of NULL)
# NULL
which(sapply(training, function(x) length(which(x=="NULL")))!=0)
# blanks
sapply(training[,names(which(sapply(training, function(x) length(which(x=="")))!=0))], function(x) length(which(x=="")))
# eRowVar <- names(which(sapply(training[,which(sapply(training, function(x) length(which(x=="")))!=0)], function(x) length(which(x=='')))<100))
# which(which(sapply(training, function(x) length(which(x=="")))!=0)<20)
# training[,which(which(sapply(training, function(x) length(which(x=="")))!=0)<20)] <- sapply(training[,names(which(which(sapply(training, function(x) length(which(x=="")))!=0)<20))], function(x) which(x==''))
# sapply(training[,names(which(sapply(training, function(x) length(which(x=="")))!=0))], function(x) length(which(x=='')))
# str(training[,names(which(sapply(training, function(x) length(which(x=="")))!=0))])
# sapply(training[,names(which(sapply(training, function(x) length(which(x=="")))!=0))], function(x) table(x, training$SPENDINGRESPONSE))
# NA
sapply(training, function(x) sum(is.na(x)))
sapply(training, function(x) sum(is.na(x)))[which(sapply(training, function(x) sum(is.na(x)))>0)]
# sapply(training[,names(which(sapply(training, function(x) sum(is.na(x)))>(nrow(training)*0.25)))], function(x) sum(is.na(x)))
# table(training$SPENDINGRESPONSE, training$f147)
# table(training$SPENDINGRESPONSE, training$f117)
# table(training$SPENDINGRESPONSE, training$f127)
# sapply(training[,names(which(sapply(training, function(x) sum(is.na(x)))>(nrow(training)*0.25)))], function(x) sum(is.na(x)))

# Check for # of unique values 
sapply(training, function(x) length(unique(x)))
sapply(training[,numVars], function(x) length(unique(x)))
sapply(training[,charVars], function(x) length(unique(x)))

# Examine response variable
table(training$SPENDINGRESPONSE) 


#########################
#########################
## DATA CLEANING AND PREPARATION

# Remove columns with large number of blanks
blankCol <- sapply(training[,names(which(sapply(training, function(x) length(which(x=="")))!=0))], function(x) length(which(x=="")))
training <- training[,names(training)[!names(training) %in% names(which(blankCol>100))]]
# Remove columns with large number of NAs
naCol <- (sapply(training, function(x) sum(is.na(x)))[which(sapply(training, function(x) sum(is.na(x)))>0)])
training <- training[,names(training)[!names(training) %in% names(which(naCol>500))]]

# Confirm no missings in character variables
sapply(training[,charVars[charVars %in% names(training)]], function(x) length(which(x=='')))
sapply(training[,charVars[charVars %in% names(training)]], function(x) length(which(is.na(x))))
charVars <- charVars[charVars %in% names(training)]
sapply(training[,charVars], function(x) length(unique(x)))
# Confirm no missings in numeric variables
sapply(training[,numVars[numVars %in% names(training)]], function(x) length(which(x=='')))
sapply(training[,numVars[numVars %in% names(training)]], function(x) length(which(is.na(x))))

# Determine which numeric variables should actually be categorical
numVars <- numVars[numVars %in% names(training)]
str(training[,names(which(sapply(training[,numVars], function(x) length(unique(x)))<1000))])
sapply(training[,names(which(sapply(training[,numVars], function(x) length(unique(x)))<1000))], function(x) length(unique(x)))
#View(head(training[,names(which(sapply(training[,numVars], function(x) length(unique(x)))<1000))],1000))
charVars <- c(charVars, "f93", "f94")
numVars <- numVars[!numVars %in% c('f93','f94', 'ID')]
#Remove ID variable (if not already removed)
training <- training[,!names(training) %in% c('ID')]

# Remove rows with lots of missing values
# sapply(training[,numVars[numVars %in% names(training)]], function(x) length(which(is.na(x))))
# View(head(training[which(is.na(training$f32)),]))
# training <- training[-which(is.na(training$f32)),]
sapply(training[,numVars[numVars %in% names(training)]], function(x) length(which(is.na(x))))
View(head(training[which(is.na(training$f4)),]))
View(training[which(training$State=="DC"),])
# f4-f9 appear to only have missing values for the state of DC
sapply(training[,c('f4', 'f5', 'f6', 'f7', 'f8')], function(x) min(x, na.rm=T))
# Since the mins are all above 0 in these columns, let's replace the NA's with zeros
for (i in (c('f4', 'f5', 'f6', 'f7', 'f8','f9'))) {
  training[which(is.na(training[,i])),i] <- 0
}
max(sapply(training[,numVars[numVars %in% names(training)]], function(x) length(which(is.na(x)))))
sapply(training[,numVars[numVars %in% names(training)]], function(x) length(which(is.na(x))))
View(head(training[which(is.na(training$f14)),]))
# No clear pattern here - remove these rows
training <- training[-which(is.na(training$f14)),]
max(sapply(training[,numVars[numVars %in% names(training)]], function(x) length(which(is.na(x)))))
max(sapply(training, function(x) length(which(is.na(x)))))

# Examine response variable
table(training$SPENDINGRESPONSE)
# Make values 0 and 1
training$SPENDINGRESPONSE <- as.numeric(sapply(training$SPENDINGRESPONSE, function(x) if (x=='Reduce National Debt and Deficit') {0} else {1}))

# Make categorical variables factors
sapply(seq(1,length(charVars)), function(x) is.factor(training[,charVars[x]]))
for (i in charVars) {
  training[,i] <- as.factor(training[,i])
}

#Trim categorical variables
catVars <- names(which(sapply(training, function(x) is.factor(x))))
sapply(training[,catVars], function(x) length(unique(x)))
training <- training[,names(training)[!names(training)%in%'f1']]
catVars <- catVars[-which(catVars=='f1')]
sapply(training[,catVars], function(x) length(unique(x)))

##### FOR CARET - gbm and glmnet ############
# Dummify factor variables for glmnet use 
trainingDummy <- dummyVars("~.",data=training, fullRank=F)
trainingDum <- as.data.frame(predict(trainingDummy,training))
print(names(trainingDum))

# Write output files (still need to clean up testing data later)
write.csv(training, 'ProjectFiles/training.csv')
write.csv(trainingDum, 'ProjectFiles/training_dum.csv')
