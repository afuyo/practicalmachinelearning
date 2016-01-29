library(caret)
#Read the data
trainUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
train<-read.csv(trainUrl)
test<-read.csv(testUrl)
#Clean the data
#Get rid of not complete cases
train <- train[, colSums(is.na(train)) == 0] 
test <- test[, colSums(is.na(test)) == 0] 
#Get Rid of not importan variables
classe <- train$classe
trainRmv <- grepl("^X|timestamp|window", names(train))
train <- train[, !trainRmv]
trainCln <- train[, sapply(train, is.numeric)]
trainCln$classe <- classe
testRmv <- grepl("^X|timestamp|window", names(test))
test <- test[, !testRmv]
testCln <- test[, sapply(test, is.numeric)]
#create training and test data sets
set.seed(3334)
inTrain <- createDataPartition(trainCln$classe, p=0.70, list=F)
trainDataSet <- trainCln[inTrain, ]
testDataSet <- trainCln[-inTrain, ]
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainDataSet, method="rf", trControl=controlRf, ntree=250)
modelRf
#wihtout cross validation
model<- randomForest(classe~.,data=trainDataSet)
model
predict1<-predict(model,testDataSet)
confusionMatrix(testData$classe,predict1)