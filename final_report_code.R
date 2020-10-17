# install packages
install.packages("ggcorrplot")
install.packages("ROCR")
library(ggplot2) # Data visualization
library(ggcorrplot)
library(class)
library(e1071)
library(kknn)
library(caret)
# import data
diabetes <- read.csv("~/Desktop/diabetes.csv")
#imputation using MICE
install.packages('mice')
library(mice)
diabetes[,2:6][diabetes[,2:6]==0] <-NA
#calculate percentage of missing data 
missing_percentage_calculated <- function(x){sum(is.na(x))/length(x)*100}
apply(diabetes,2,missing_percentage_calculated)
methods(mice)
processedData <- mice(diabetes[, !names(diabetes) %in% "Outcome"],m=5,maxit=5,meth='norm.predict',seed=500)
#Clean data by imputation
data <- complete(processedData,2)
#Add Outcome back
data$Outcome <- diabetes$Outcome
View(data)


# correlation data
set.seed(1)
data_cor <- round(cor(data[1:8]),1)
ggcorrplot(data_cor)

# scale data
set.seed(1)
data.scale = as.data.frame(scale(data[,-9]))
data.scale$Outcome = data$Outcome

# Split dataset into train and test sets
set.seed(1)
train_index <- sample(1:nrow(data.scale), 0.8 * nrow(data.scale))
test_index <- setdiff(1:nrow(data.scale), train_index)
#Build X_train, y_train, X_test, y_test
data.scale$Outcome = as.factor(data.scale$Outcome)
set.seed(1)
train <- data.scale[train_index,]
x_train <- data.scale[train_index, -9]
y_train <- data.scale[train_index, "Outcome"]

set.seed(1)
test <- data.scale[-train_index,]
x_test <- data.scale[test_index, -9]
y_test <- data.scale[test_index, "Outcome"]

#KNN modelling
for (i in 1:20){
  set.seed(1)
  knn.pred <- knn(train=x_train, test=x_test, cl=y_train, k=i)
  print(paste("K = ",i, ": ",(mean(knn.pred!=y_test))))
}

set.seed(1)
knn.pred <- knn(train=x_train, test=x_test, cl=y_train, k=7)
table(knn.pred, y_test)

grid1 <- expand.grid(.k=seq(2,20, by=1))
control <- trainControl(method="cv")
set.seed(1)
KNN <- train(Outcome~., data=data.scale, method="knn", 
             trControl=control, tuneGrid=grid1)
KNN
plot(KNN)

KNN.obj <- knn(train = data.scale[1:8], test = data.scale[1:8],
               cl=data.scale$Outcome, k=3)
mean(KNN.obj != data.scale$Outcome)
table(KNN.obj, data.scale$Outcome)
summary(KNN.obj)

for (i in 2:20){
  set.seed(1)
  KNN_forloop <- knn(train = data.scale[1:8], test = data.scale[1:8],
                     cl=data.scale$Outcome, k=i)
  print(paste("K = ",i, ": ",(mean(KNN_forloop==data.scale$Outcome))))
}
table(KNN.obj, data.scale$Outcome)



# Train scale data set, exclude column 9 - diabetes output
x <- scale(data)
plot(x)
library(factoextra)

k.max <- 10
wss <- numeric(k.max)

for (k in 1:k.max){
  wss[k] <- eclust(x,
                   FUNcluster = "kmeans",
                   k=k,
                   nstart=50)$tot.withinss
}
plot(wss,
     type="b", pch=19, col=2,
     main = "Total Within-Cluster Sum of Squares, for K=1,...,10",
     xlab = "Number of Clusters K", ylab = "Total WSS")


fviz_nbclust(x, kmeans, nstart = 50, method = "gap_stat", nboot = 50)



ec.obj <- eclust(x,
                 FUNcluster="kmeans", k = 2,
                 nstart = 50)
ec.obj$silinfo
ec.obj$gap_stat

k.max <- 10
silh.coef <- numeric(k.max)
for (k in 2:10){
  silh.coef[k] <- eclust(x,
                         FUNcluster="kmeans",
                         k = k,
                         graph=0,
                         nstart = 50)$silinfo$avg.width
}
plot (silh.coef,
      type="b", pch=19, col=4)

diabetes <- read.csv("diabetes.csv")
x <- diabetes
eclust (x, FUNcluster = "kmeans", k=2, nstart =50)$tot.withinss



#svm
#Using the SVM model defined in the package with all variables considered in building the model
svm_model=svm(Outcome~.,data=train,type='C-classification')
#Summary will list the respective parameters uch as cost, gamma, etc.
summary(svm_model)
#Predicting the data with the input to be the dataset itself, we can calculate the accuracy with a confusion matrix
pred_train=predict(svm_model,newdata=train)
table(pred_train,train$Outcome)
# Test error with svm 
pred_test = predict(svm_model, newdata = test)
table(pred_test,test$Outcome)
#Now let's tune the SVM parameters to get a better accuracy on the training dataset
svm_tune_radial <- tune(svm, Outcome~.,data=train,kernel="radial", ranges = list(cost=c(0.001,0.1,1,5,10,100),gamma=c(0.5,1,2,3,4,5)))
summary(svm_tune_radial)
svm_tune_linear <- tune(svm, Outcome~.,data=train,kernel="linear", ranges = list(cost=c(0.001,0.1,1,5,10,100)))
summary(svm_tune_linear)
svm_tune_polynomial=tune(method=svm,Outcome~.,data=train,kernel="polynomial",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100), degree=c(2,3,4)))
summary(svm_tune_polynomial)
#Gives an optimal cost to be 5 
svm_model_after_tune <- svm(Outcome ~ ., data=train, type='C-classification',kernel="radial", cost=1, gamma=0.5)
summary(svm_model_after_tune)

#The results show us that there is an improved accuracy of about 98%, results are obtained in the form of a confusion matrix
pred <- predict(svm_model_after_tune,test)
table(pred,test$Outcome)



