library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(corrplot)
set.seed(30001)
# reading the original data
train_Raw <- read.csv('./pml-training.csv', header=T, na.strings = c("", "NA"))
validationRaw <- read.csv('./pml-testing.csv', header=T, na.strings = c("", "NA"))
dim(train_Raw)
dim(validationRaw)

# data cleanup
#Data cleanup is a very important step in this analysis. here we will get rid of the observations with missing values and some meaningless variables. 

sum(complete.cases(train_Raw))

# first we remove columns that contain NA/missing values

train_Raw <- train_Raw[, colSums(is.na(train_Raw)) == 0] 
validationRaw <- validationRaw[, colSums(is.na(validationRaw)) == 0] 

# Since data doesn't have time dependence so the columns with time information are removed from tha dataset including the first column (i.e. observation number). However, in the cleaned dataset, the variable 'classe' is kept in the dataset.

train_Raw <- train_Raw[,c(8:60)]
validationRaw<- validationRaw[, c(8:60)]

# Partitioning the train data into two parts
train_sample <- createDataPartition(y=train_Raw$classe, p=0.7, list=FALSE)
train_data <- train_Raw[train_sample, ]
test_data <- train_Raw[-train_sample, ]

# Data Modeling
# In this analysis we tried to fit Random forest algorithm to fit predictive model for recognition of activity.
# The reasons for selecting Random Forest are:
# It automatically identifies the important variables and
# It produces robust correlated covariates and outliers
# While employing Random Forest algorithm we applied 5 fold cross validation of the algorithm.

Rfcontrol <- trainControl(method="cv", 5) # Specifying Random Forest method

Rfmodel <- train(classe ~ ., data=train_data, method="rf", trControl=Rfcontrol, ntree=250)
Rfmodel #display the model parameters

# After building model, the performance was tested using the partitioned test data as below:

Rfpredict <- predict(Rfmodel, test_data)
confusionMatrix(test_data$classe, Rfpredict)



#Determining model accuracy and Out-of-sample Error Estimation

accuracy <- postResample(Rfpredict, test_data$classe) # Model Accuracy
accuracy
oose <- 1 - as.numeric(confusionMatrix(test_data$classe, Rfpredict)$overall[1]) # Out-of-sample error estimation
oose

# From the above parameter, the accuracy of modeling is estimated as 99.42% and out-of-sample error is 0.58%

# After analysing the accuracy and out-of-sample error it was decided to go ahead Random Forest model to predict the parameters 'classe' in the validation dataset. Predicting the the manner in which excercise was carried using the validation dataset. In this step we apply the model to the original test dataset as downloaded from the source.
# After reviewing data it was noticed a column called "problem_id" exist which needs to be removed before prediction.

validationRaw<-validationRaw[,c(1:53)]

Finalresult <- predict(Rfmodel, validationRaw)
Finalresult

# Viewing the correlation matrix

CP <- cor(train_data[, -length(names(train_data))])
corrplot(CP, method="circle")

#Viewing Decision tree

treeModel <- rpart(classe ~ ., data=train_data)
prp(treeModel) # fast plot
