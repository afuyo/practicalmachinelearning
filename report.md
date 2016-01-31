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
In random forests, there is no need for cross-validation . It is estimated internally, during the run, as follows:

Each tree is constructed using a different bootstrap sample from the original data. About one-third of the cases are left out of the bootstrap sample and not used in the construction of the kth tree.   

```r
modelRf <- train(classe ~ ., data=trainDataSet, method="rf", ntree=250)
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
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9882025  0.9850716  0.001694686  0.002134850
##   27    0.9882897  0.9851828  0.001714376  0.002162997
##   52    0.9791543  0.9736255  0.004128074  0.005208552
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
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
## roll_belt            100.000
## pitch_forearm         55.485
## yaw_belt              53.893
## pitch_belt            44.559
## roll_forearm          43.495
## magnet_dumbbell_z     43.157
## magnet_dumbbell_y     42.898
## accel_dumbbell_y      22.257
## magnet_dumbbell_x     17.015
## accel_forearm_x       16.356
## magnet_belt_z         16.172
## roll_dumbbell         15.988
## accel_belt_z          15.120
## magnet_forearm_z      13.903
## accel_dumbbell_z      13.713
## total_accel_dumbbell  13.151
## gyros_belt_z          10.634
## yaw_arm               10.451
## magnet_belt_y         10.219
## magnet_belt_x          9.334
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
##          B   10 1126    3    0    0
##          C    0    7 1014    5    0
##          D    0    0    3  960    1
##          E    0    0    3    1 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9921, 0.9961)
##     No Information Rate : 0.2862          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9941   0.9938   0.9912   0.9938   0.9991
## Specificity            1.0000   0.9973   0.9975   0.9992   0.9992
## Pos Pred Value         1.0000   0.9886   0.9883   0.9959   0.9963
## Neg Pred Value         0.9976   0.9985   0.9981   0.9988   0.9998
## Prevalence             0.2862   0.1925   0.1738   0.1641   0.1833
## Detection Rate         0.2845   0.1913   0.1723   0.1631   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9970   0.9955   0.9944   0.9965   0.9991
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
##          A 1659    6    8    1    0
##          B   14 1103   19    3    0
##          C    1    7 1011    7    0
##          D    0    2   12  947    3
##          E    3    3    6    2 1068
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9835          
##                  95% CI : (0.9799, 0.9866)
##     No Information Rate : 0.285           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9792          
##  Mcnemar's Test P-Value : 0.001294        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9893   0.9839   0.9574   0.9865   0.9972
## Specificity            0.9964   0.9924   0.9969   0.9965   0.9971
## Pos Pred Value         0.9910   0.9684   0.9854   0.9824   0.9871
## Neg Pred Value         0.9957   0.9962   0.9907   0.9974   0.9994
## Prevalence             0.2850   0.1905   0.1794   0.1631   0.1820
## Detection Rate         0.2819   0.1874   0.1718   0.1609   0.1815
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9929   0.9882   0.9771   0.9915   0.9971
```

```r
oose <- 1 - as.numeric(confusionMatrix(testReduced$classe,predReduced)$overall[1])
oose
```

```
## [1] 0.01648258
```

Out of sample error

```r
oose <- 1 - as.numeric(confusionMatrix(testReduced$classe,predReduced)$overall[1])
oose
```

```
## [1] 0.01648258
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

