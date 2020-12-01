## Code for capstone project Capstone2.pdf
## Written by Diana Ojeda-Revah

# This file includes all the code used for the capstone Capstone2.pdf

########################
#      load libraries
#######################
if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)
if (!require(caret)) install.packages('caret')
library(caret)
if (!require(ggridges)) install.packages('ggridges')
library(ggridges)
if (!require(corrplot)) install.packages('corrplot')
library(corrplot)
if (!require(rpart)) install.packages('rpart')
library(rpart)
if (!require(randomForest)) install.packages('randomForest')
library(randomForest)
if (!require(gridExtra)) install.packages('gridExtra')
library(gridExtra)
if (!require(factoextra)) install.packages('factoextra')
library(factoextra)



#color preferences
myfill = "slategray3"
mycolor = "grey27"



####################################
#      Download and label data
################################
# Download, label and save data in subdirectory

if(!dir.exists("data"))
    dir.create("data")

# url where data is located
urldata <-"https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/segment/segment.dat"

#read data
segment <- read.table(urldata, header=FALSE, sep="")

# Name variables
var_names <-  c("reg_cent_col","reg_cent_row","reg_pxl_ct", "shrt_ln_dns_5", "shrt_ln_dns_2", "vedge_mn", "vegde_sd", "hedge_mn", "hedge_sd", "int_mn", "rawred_mn", "rawblue_mn", "rawgreen_mn", "exred_mn", "exblue_mn", "exgreen_mn", "value_mn", "sat_mn", "hue_mean","segment")

colnames(segment) <- var_names

#label classes
labl <- c("brickface", "sky", "foliage", "cement", "window", "path", "grass")

segment <-segment %>% mutate(segment = factor(segment, labels=labl)) 



save(segment, file="data/segment.RData")

# Load data

load("data/segment.RData")

#################################
#        Data exploration       #
#################################

# Number of instances and variables
# find if data is missing
# names of variables
dim(segment)
any(is.na(segment))
names(segment)

#function to find summary statistics
sts <- function(mydf) {
    sapply( mydf , function(x) rbind(   mean = mean(x) ,
                                        sd = sd(x) ,
                                        median = median(x) ,
                                        minimum = min(x) ,
                                        maximum = max(x)  ))
}



#create table of summary statistics
sumstats <-data.frame(sts(segment[,1:19])) %>% 
    mutate_all(function(x) round(x,digits=2))   
rownames(sumstats) = c("mean", "sd", "median", "minimum", "maximum")

#summary statistics of predictors
sumstats

# find variables with near zero variance and inspect them
nzv(segment)
table(segment$reg_pxl_ct)
table(segment$shrt_ln_dns_5)

tabl <-table(segment$shrt_ln_dns_2, segment$segment)


# Drop variables with non zero variance
segment <- segment %>% select(-reg_pxl_ct, -shrt_ln_dns_5, -shrt_ln_dns_2)

#histograms

#histograms for each of the remaining predictors.

segment %>%
    keep(is.numeric) %>% 
    gather() %>% 
    ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram(bins=30, col= mycolor, fill=myfill)

# density plots for each predictor by segment class. The command used 
# geom_denisity_ridges() uses the package ggridges

segment %>%
    pivot_longer(!segment, names_to="variables", values_to ="values" ) %>% 
    ggplot(aes(values, segment, fill=segment)) +
    geom_density_ridges()+
    facet_wrap(~ variables, scales = "free") +
    scale_y_discrete(labels=NULL)


#boxplots for each variable against class variable (segment)
# It is done in two parts to have a reasonably sized plots

segment %>% 
    select(segment,1:6) %>%
    pivot_longer(!segment, names_to="variables", values_to ="values" ) %>%
    mutate(variables=factor(variables)) %>%
    ggplot(aes(segment, values)) +
    geom_boxplot(fill=myfill) +
    facet_wrap(~variables, scales="free")+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))


segment %>% 
    select(segment,c(7,11,12,13,14,15)) %>%
    pivot_longer(!segment, names_to="variables", values_to ="values" ) %>%
    mutate(variables=factor(variables)) %>%
    ggplot(aes(segment, values)) +
    geom_boxplot(fill=myfill) +
    facet_wrap(~variables, scales="free")+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))




# find correlation matrix

corel <-segment %>% select(-segment) %>%
    cor()

#correlation plot using library corrplot
corrplot(corel)

#find programatically highly correlated and linear combinations
#using caret

findCorrelation(corel)
findLinearCombos(corel)

#############################
#     Methods and analysis
############################

# reload the data. The data was altered for the exploratory part and needs to be
# reloaded for the analysis.

load("data/segment.RData")

#remove the non zero variance variables
segment <- segment %>% select(-reg_pxl_ct, -shrt_ln_dns_5, -shrt_ln_dns_2)



####### train/test split data

set.seed(100)
trainIndex <- createDataPartition(segment$segment,
                                  p=0.8,
                                  times=1,
                                  list=F)
segment_train <- segment[trainIndex,]
segment_test <- segment[trainIndex,]


##############################################
# Create 3 different pairs of test/train 
# sets with different preprocesses to compare
# how each preprocess procedure affects results
#############################################3

## Preprocess #1
## Remove 3 of the 4 highly correlated variables

train_inc <- segment_train %>% select(-rawred_mn, -rawblue_mn, -rawgreen_mn, -value_mn)
test_inc <- segment_test %>% select(-rawred_mn, -rawblue_mn, -rawgreen_mn, -value_mn)

## Preprocess #2
## Center and scale data that had correlated variables removed
cent_scale <- preProcess(test_inc, method = c("center", "scale"))
train_cs <- predict(cent_scale, train_inc)
test_cs <- predict(cent_scale, test_inc)

## Preprocess #3
##Principal components 

#Analyze principal components
# perform principal components to study the scree plot
pcomp <- prcomp(segment_train[,-17], center=TRUE, scale=TRUE)

#scree plot
fviz_eig((pcomp), addlabels = TRUE, ylim = c(0, 50))

#contribution of variables to the first principal component
fviz_cos2(pcomp, choice = "var", axes = 1)

# Preprocess with principal components, using caret
# The transformation is done to all the data, including highly
# correlated predictors. The preprocess automatically centers and scales
prep <-preProcess(segment_train, method="pca", thresh=0.95)

# Find out how many principal components are kept
prep

# train/test sets with transformed variables
train_pca <-predict(prep,segment_train)
test_pca <-  predict(prep, segment_test)

#####################################################
# Model fitting with dataset with transformation #1
####################################################


#############     knn (k nearest neighbors)


#train model on a grid of different values for k, for training data.
set.seed(100)
train_knn <- train(segment ~ .,
                   method = "knn",
                   data = train_inc,
                   tuneGrid = data.frame(k = seq(2, 50, 2)))

#plot the results of bootsrap on each k
ggplot(train_knn, highlight = TRUE)


# accuracy and confusion matrix
# accuracies are assigned to variables to compare them at the end
accu_knn1 <-confusionMatrix(predict(train_knn, test_inc, type = "raw"),
                           test_inc$segment)$overall["Accuracy"]

cf <-confusionMatrix(predict(train_knn, test_inc, type = "raw"),
                test_inc$segment)$table

#number of missclassified instances.
misclas <-(1-accu_knn1[1])*nrow(test_inc)


# Explore some examples of where the missclassifications are.
# create data.frame with missclassifications to use for
#plotting.
# Test set gets a new column of predicted labels and only the
# Mislabeled are filtered.

missed <-test_inc %>% 
    mutate(predicted = predict(train_knn, test_inc, type = "raw")) %>%
    mutate(real= factor(ifelse(segment==predicted, 0,1))) %>%
    filter(real==1)

#Boxplots of two predictors as function of each class in test set, showing 
# where the misclassified points are.

p1 <-test_inc %>% 
    ggplot() + aes(segment,reg_cent_row) + geom_boxplot(aes(color=segment)) + 
    geom_text(data=missed, aes( label=predicted), alpha=0.5)+
    theme(legend.position = "none")

p2 <- test_inc %>% 
    ggplot() + aes(segment,exblue_mn) + geom_boxplot(aes(color=segment)) + 
    geom_text(data=missed, aes( label=predicted), alpha=0.5)+
    theme(legend.position = "none")

grid.arrange(p1,p2, ncol=1)


######### Classification tree

# fit tree to training data
set.seed(101)
train_rpart <- train(segment ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = train_inc)

#plot bootstrap results
plot(train_rpart)

# Plot actual tree, with best tune
plot(train_rpart$finalModel,margin=0.1)
text(train_rpart$finalModel, cex = 0.75)

train_rpart$bestTune

#accuracy with test data
accu_rpart1 <-confusionMatrix(predict(train_rpart, test_inc, type = "raw"),
                             test_inc$segment)$overall["Accuracy"]
accu_rpart1



################# RANDOM FOREST
#we will time the process for comparison

set.seed(10)
st <- Sys.time()
#mtry is the number of variables from which the algorithm is allowed to choose
# each split
grid <- data.frame(mtry = 1:12)
control <- trainControl(method="cv", number = 5)

# Train random forest in training data
train_rf <- train(segment~.,
                  method = "rf",
                  ntree = 250,
                  tuneGrid = grid,
                  trControl = control,
                  data = train_inc)

end <- Sys.time()

#running time
time_rf <-  end-st

# plot
ggplot(train_rf)
train_rf$bestTune

## predicions
predictions <- predict(train_rf, newdata=test_inc)

# Confusion matrix and accuracy
cftree <-confusionMatrix(predictions, test_inc$segment)
accu_rf1 <- cftree$overall["Accuracy"]

#variable importance plot
varImpPlot(train_rf$finalModel)


###############################################################
# Model fitting with dataset with transformation #2: center-scale
################################################################


######knn on centered-scaled data


#train model on a grid of different values for k, for center-scaled train data
set.seed(100)
train_knn2 <- train(segment ~ .,
                   method = "knn",
                   data = train_cs,
                   tuneGrid = data.frame(k = seq(2, 50, 2)))

#number of neighbors for best tune

train_knn2$bestTune


#accuracy 
accu_knn2 <-confusionMatrix(predict(train_knn2, test_cs, type = "raw"),
                            test_cs$segment)$overall["Accuracy"]



#number of missclassified instances.
misclas2 <-(1-accu_knn2[1])*nrow(test_cs)

############ Classification tree model on center-scale

set.seed(101)
#train classification tree with center-scale train data
train_rpart2 <- train(segment ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = train_cs)
plot(train_rpart2)

#best tune
train_rpart2$bestTune

#confusion matrix on center-scaled test set
confusionMatrix(predict(train_rpart, test_cs$test, type = "raw"),
                test_cs$segment)$byClass
#accuracy
accu_rpart2 <-confusionMatrix(predict(train_rpart2, test_cs$test, type = "raw"),
                             test_cs$segment)$overall["Accuracy"]
accu_rpart2


################# RANDOM FOREST with center-scale
#we will time the process for comparison

set.seed(10)
st <- Sys.time()

#mtry is the number of variables from which the algorithm is allowed to choose
#for each split
#Fit random forest to center-scaled train data
grid <- data.frame(mtry = 1:12)
control <- trainControl(method="cv", number = 5)
train_rf2 <- train(segment~.,
                  method = "rf",
                  ntree = 250,
                  tuneGrid = grid,
                  trControl = control,
                  data = train_cs)

end <- Sys.time()
#running time
time_rf2 <-  end-st

#best tune
train_rf2$bestTune

## predicions and accuracy on test set
predictions <- predict(train_rf2, newdata=test_cs)
# confusion matrix
cftree2 <-confusionMatrix(predictions, test_cs$segment)
#accuracy
accu_rf2 <- cftree$overall["Accuracy"]


########################################################
# Model fitting with dataset with transformation #3:PCA
#######################################################

#########knn on pca scores


#train model on a grid of different values for k for PCA train data
set.seed(100)
train_knn3 <- train(segment ~ .,
                    method = "knn",
                    data = train_pca,
                    tuneGrid = data.frame(k = seq(2, 50, 2)))

#number of neighbors for best tune

train_knn3$bestTune

#accuracy 
accu_knn3 <-confusionMatrix(predict(train_knn3, test_pca, type = "raw"),
                            test_pca$segment)$overall["Accuracy"]



######## Classification tree model on PCA

set.seed(101)
#model training
train_rpart3 <- train(segment ~ .,
                      method = "rpart",
                      tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                      data = train_pca)
#find best tune
plot(train_rpart3)
train_rpart3$bestTune

#confusion matrix
confusionMatrix(predict(train_rpart3, test_pca$test, type = "raw"),
                test_pca$segment)$byClass

#accuracy, assign it to a variable to make a table
accu_rpart3 <-confusionMatrix(predict(train_rpart3, test_pca$test, type = "raw"),
                              test_pca$segment)$overall["Accuracy"]
accu_rpart3


############ RANDOM FOREST with PCA
#we will time the process for comparison

set.seed(10)
st <- Sys.time()
#mtry is the number of variables from which the algorithm chooses each
#split. It is lower than in previous cases because we only have 8 Principal
# components
grid <- data.frame(mtry = 1:8)
# set control to cross-validation
control <- trainControl(method="cv", number = 5)

#train on transformed TRAINING set
train_rf3 <- train(segment~.,
                   method = "rf",
                   ntree = 250,
                   tuneGrid = grid,
                   trControl = control,
                   data = train_pca)

end <- Sys.time()
#find time
time_rf3 <-  end-st
#find best tune
train_rf3$bestTune

## predicions and accuracy on transformed test set
predictions <- predict(train_rf3, newdata=test_pca)

cftree3 <-confusionMatrix(predictions, test_pca$segment)
accu_rf3 <- cftree$overall["Accuracy"]
#variable importance
varImp(train_rf3)
#Number of missclassified
(1-accu_rf3[1])*nrow(test_inc)

# table of results (accuracy), including times for rf

res_accuracy <- data.frame(knn = c(accu_knn1, accu_knn2, accu_knn3),
                       tree = c(accu_rpart1, accu_rpart2, accu_rpart3),
                       Random_forest= c(accu_rf1, accu_rf2, accu_rf3),
                       RF_time = c(time_rf, time_rf2, time_rf3))

row.names(res_accuracy) = (c("Highly correlated remove", "Center-scale", "PCA"))
