# Practical Machine Learning
Artur Mrozowski  
28. jan. 2016  

### Introduction  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.  



```
## Warning: package 'caret' was built under R version 3.2.3
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.2.2
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.3
```

```
## Warning: package 'rpart' was built under R version 3.2.3
```

```
## Warning: package 'rpart.plot' was built under R version 3.2.3
```

## Read the data


```r
trainUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
train<-read.csv(trainUrl)
test<-read.csv(testUrl)
```
##Clean the data
The columns that contain NA values need to be removed. Also the columms that don't make much sensne in the prediction model and create unecessary noise.

```r
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
```
##Create training and test data sets
By splitting the data into training(70%) and validation(30%) we'll be able to validate the model.  

```r
set.seed(3334)
inTrain <- createDataPartition(trainCln$classe, p=0.70, list=F)
trainDataSet <- trainCln[inTrain, ]
testDataSet <- trainCln[-inTrain, ]
```
#Train the model randomforest
We will use **Random Forest** algorithm

In random forests each tree is constructed using a different bootstrap sample from the original data. About one-third of the cases are left out of the bootstrap sample and not used in the construction of the kth tree.
We will use **5-fold cross validation** when applying the algorithm.

```r
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainDataSet, method="rf",trControl=controlRf, ntree=250)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.2.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
modelRf
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10991, 10990, 10988, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9906821  0.9882117  0.002632884  0.003332714
##   27    0.9901723  0.9875668  0.002820579  0.003570059
##   52    0.9829646  0.9784479  0.003885841  0.004917619
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```
#Variable importance

```r
varImp(modelRf)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt             100.00
## yaw_belt               79.21
## magnet_dumbbell_z      65.89
## magnet_dumbbell_y      60.70
## pitch_belt             60.21
## pitch_forearm          57.37
## magnet_dumbbell_x      54.21
## roll_forearm           49.95
## accel_dumbbell_y       43.91
## accel_belt_z           41.20
## magnet_belt_y          41.19
## magnet_belt_z          40.97
## roll_dumbbell          39.93
## accel_dumbbell_z       35.18
## roll_arm               31.86
## accel_forearm_x        29.77
## accel_arm_x            28.29
## total_accel_dumbbell   28.22
## accel_dumbbell_x       27.94
## gyros_belt_z           27.50
```

```r
ggplot(varImp(modelRf))
```

![](report_files/figure-html/unnamed-chunk-6-1.png)

we can see that rooll_belt and yaw_belt  are by far the most important variables

#accuracy plot

```r
plot.train(modelRf)
```

![](report_files/figure-html/unnamed-chunk-7-1.png)

Accuracy is 99,4%

Let's evaluate model results in confusion matrix after estimating the performance on the validation data set

```r
predRf<-predict(modelRf,testDataSet)
confusionMatrix(testDataSet$classe,predRf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    7 1129    3    0    0
##          C    0    8 1017    1    0
##          D    0    0   14  947    3
##          E    0    0    1    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9937          
##                  95% CI : (0.9913, 0.9956)
##     No Information Rate : 0.2856          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.992           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9930   0.9826   0.9989   0.9972
## Specificity            1.0000   0.9979   0.9981   0.9966   0.9998
## Pos Pred Value         1.0000   0.9912   0.9912   0.9824   0.9991
## Neg Pred Value         0.9983   0.9983   0.9963   0.9998   0.9994
## Prevalence             0.2856   0.1932   0.1759   0.1611   0.1842
## Detection Rate         0.2845   0.1918   0.1728   0.1609   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9979   0.9954   0.9904   0.9978   0.9985
```

Let's reduce number of variable to the most importan ones. Let's reduce training and testing set accordingly and validation set on which we make the predictions for the classe variable.

```r
trainReduced<-trainDataSet[,c("roll_belt","yaw_belt","magnet_dumbbell_z","pitch_belt","magnet_dumbbell_y","magnet_dumbbell_x","pitch_forearm","classe")]
testReduced<-testDataSet[,c("roll_belt","yaw_belt","magnet_dumbbell_z","pitch_belt","magnet_dumbbell_y","magnet_dumbbell_x","pitch_forearm","classe")]
testClnReduced<-testCln[,c("roll_belt","yaw_belt","magnet_dumbbell_z","pitch_belt","magnet_dumbbell_y","magnet_dumbbell_x","pitch_forearm")]
```

Now let's train the model on the reduced data set.

```r
modelRfReduced <- train(classe ~ ., data=trainReduced, method="rf", ntree=250)
```
Prediciton for the new model with reduced number of variables.

```r
predReduced<-predict(modelRfReduced,testReduced)
confusionMatrix(testReduced$classe,predReduced)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1656    6   10    2    0
##          B   15 1100   20    4    0
##          C    2    9 1007    8    0
##          D    1    1   14  944    4
##          E    3    4    5    3 1067
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9811          
##                  95% CI : (0.9773, 0.9845)
##     No Information Rate : 0.285           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9761          
##  Mcnemar's Test P-Value : 0.001124        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9875   0.9821   0.9536   0.9823   0.9963
## Specificity            0.9957   0.9918   0.9961   0.9959   0.9969
## Pos Pred Value         0.9892   0.9658   0.9815   0.9793   0.9861
## Neg Pred Value         0.9950   0.9958   0.9899   0.9965   0.9992
## Prevalence             0.2850   0.1903   0.1794   0.1633   0.1820
## Detection Rate         0.2814   0.1869   0.1711   0.1604   0.1813
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9916   0.9870   0.9748   0.9891   0.9966
```

```r
oose <- 1 - as.numeric(confusionMatrix(testReduced$classe,predReduced)$overall[1])
oose
```

```
## [1] 0.01886151
```

Out of sample error

```r
oose <- 1 - as.numeric(confusionMatrix(testReduced$classe,predReduced)$overall[1])
oose
```

```
## [1] 0.01886151
```


#Results


```r
resReduced<-predict(modelRfReduced,testClnReduced)
resReduced
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

#Visualizaiton of the tree model

```r
treeModel <- rpart(classe ~ ., data=trainReduced, method="class")
 prp(treeModel)
```

![](report_files/figure-html/unnamed-chunk-14-1.png)

