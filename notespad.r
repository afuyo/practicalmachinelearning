library(caret)
trainUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
train<-read.csv(trainUrl)
test<-read.csv(testUrl)
train <- train[, colSums(is.na(train)) == 0] 
test <- test[, colSums(is.na(test)) == 0] 
classe <- train$classe
trainRemove <- grepl("^X|timestamp|window", names(train))
train <- train[, !trainRemove]
trainCleaned <- train[, sapply(train, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(test))
test <- test[, !testRemove]
testCleaned <- test[, sapply(test, is.numeric)]
