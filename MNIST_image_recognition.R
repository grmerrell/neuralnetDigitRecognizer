# Neural Network classification of MNIST data
# retrieved CSV from http://pjreddie.com/projects/mnist-in-csv/ 
# install.packages("EBImage") <- useful package for sifting through color images
# can classify pixels as 3 separate matrices of RGB info as well as place
# filters over top of image, greyscale format image

library(nnet)
library(neuralnet)
library(caret)
library(RSNNS)
library(dplyr)
library(car)
library(e1071)
library(ROCR)


train <- read.csv(file = 'C:/Users/greg.merrell/Downloads/mnist_train.csv', header = FALSE)
test <- read.csv(file = 'C:/Users/greg.merrell/Downloads/mnist_test.csv', header = FALSE)

colnames(train) <- c("Numbers", 1:784)
colnames(test) <- c("Numbers", 1:784)

# function that'll show what digit a given row of pixels shows
displayDigit <- function(X){ 
  m <- matrix(unlist(X), nrow = 28, byrow = T)
  # unpacks X with 28 rows (we have 28 by 28 pixel images)
  # byrow = TRUE fills the matrix by rows
  m <- t(apply(m, 2, rev)) 
  # t() swaps columns for rows
  image(m, col=grey.colors(255))
  # image() shows the 0-255 grey scale of each pixel
}
par(mfrow = c(1,1))
displayDigit(train[10,-1]) # displays all pixel info other than number
train[10,1] # and here's checking to see it's the same number
displayDigit(train[3,-1])


# now we're going to need to split up the numbers from the numbers
# column into binary encoding so that we'll have 10 output neurons
# the reason why we're doing this is so that our activation functions
# we'll be firing on those output neurons when they're a given digit
# you could encode this into four output neurons (2^4 = 16), but
# this way the neural net will focus in on
# an output neuron being a specific number and firing
# rather than the four most significant parts of an image that
# make up a given digit in combination with the other output neurons
# in order for it to "fire"

dim(train)
for (i in 1:length(train$Numbers)) {
  if (train$Numbers[i] == 0){
    train$Digit0[i] <- 1
    train$Digit1[i] <- 0
    train$Digit2[i] <- 0
    train$Digit3[i] <- 0
    train$Digit4[i] <- 0
    train$Digit5[i] <- 0
    train$Digit6[i] <- 0
    train$Digit7[i] <- 0
    train$Digit8[i] <- 0
    train$Digit9[i] <- 0
  }
  else if (train$Numbers[i] == 1) {
    train$Digit0[i] <- 0
    train$Digit1[i] <- 1
    train$Digit2[i] <- 0
    train$Digit3[i] <- 0
    train$Digit4[i] <- 0
    train$Digit5[i] <- 0
    train$Digit6[i] <- 0
    train$Digit7[i] <- 0
    train$Digit8[i] <- 0
    train$Digit9[i] <- 0
  }
  else if (train$Numbers[i] == 2){
    train$Digit0[i] <- 0
    train$Digit1[i] <- 0
    train$Digit2[i] <- 1
    train$Digit3[i] <- 0
    train$Digit4[i] <- 0
    train$Digit5[i] <- 0
    train$Digit6[i] <- 0
    train$Digit7[i] <- 0
    train$Digit8[i] <- 0
    train$Digit9[i] <- 0
  }
  else if (train$Numbers[i] == 3){
    train$Digit0[i] <- 0
    train$Digit1[i] <- 0
    train$Digit2[i] <- 0
    train$Digit3[i] <- 1
    train$Digit4[i] <- 0
    train$Digit5[i] <- 0
    train$Digit6[i] <- 0
    train$Digit7[i] <- 0
    train$Digit8[i] <- 0
    train$Digit9[i] <- 0
  }
  else if (train$Numbers[i] == 4){
    train$Digit0[i] <- 0
    train$Digit1[i] <- 0
    train$Digit2[i] <- 0
    train$Digit3[i] <- 0
    train$Digit4[i] <- 1
    train$Digit5[i] <- 0
    train$Digit6[i] <- 0
    train$Digit7[i] <- 0
    train$Digit8[i] <- 0
    train$Digit9[i] <- 0
  }
  else if (train$Numbers[i] == 5){
    train$Digit0[i] <- 0
    train$Digit1[i] <- 0
    train$Digit2[i] <- 0
    train$Digit3[i] <- 0
    train$Digit4[i] <- 0
    train$Digit5[i] <- 1
    train$Digit6[i] <- 0
    train$Digit7[i] <- 0
    train$Digit8[i] <- 0
    train$Digit9[i] <- 0
  }
  else if (train$Numbers[i] == 6){
    train$Digit0[i] <- 0
    train$Digit1[i] <- 0
    train$Digit2[i] <- 0
    train$Digit3[i] <- 0
    train$Digit4[i] <- 0
    train$Digit5[i] <- 0
    train$Digit6[i] <- 1
    train$Digit7[i] <- 0
    train$Digit8[i] <- 0
    train$Digit9[i] <- 0
  }
  else if (train$Numbers[i] == 7){
    train$Digit0[i] <- 0
    train$Digit1[i] <- 0
    train$Digit2[i] <- 0
    train$Digit3[i] <- 0
    train$Digit4[i] <- 0
    train$Digit5[i] <- 0
    train$Digit6[i] <- 0
    train$Digit7[i] <- 1
    train$Digit8[i] <- 0
    train$Digit9[i] <- 0
  }
  else if (train$Numbers[i] == 8){
    train$Digit0[i] <- 0
    train$Digit1[i] <- 0
    train$Digit2[i] <- 0
    train$Digit3[i] <- 0
    train$Digit4[i] <- 0
    train$Digit5[i] <- 0
    train$Digit6[i] <- 0
    train$Digit7[i] <- 0
    train$Digit8[i] <- 1
    train$Digit9[i] <- 0
  }
  else {
    train$Digit0[i] <- 0
    train$Digit1[i] <- 0
    train$Digit2[i] <- 0
    train$Digit3[i] <- 0
    train$Digit4[i] <- 0
    train$Digit5[i] <- 0
    train$Digit6[i] <- 0
    train$Digit7[i] <- 0
    train$Digit8[i] <- 0
    train$Digit9[i] <- 1
  }
}

for (i in 1:length(test$Numbers)) {
  if (test$Numbers[i] == 0){
    test$Digit0[i] <- 1
    test$Digit1[i] <- 0
    test$Digit2[i] <- 0
    test$Digit3[i] <- 0
    test$Digit4[i] <- 0
    test$Digit5[i] <- 0
    test$Digit6[i] <- 0
    test$Digit7[i] <- 0
    test$Digit8[i] <- 0
    test$Digit9[i] <- 0
  }
  else if (test$Numbers[i] == 1) {
    test$Digit0[i] <- 0
    test$Digit1[i] <- 1
    test$Digit2[i] <- 0
    test$Digit3[i] <- 0
    test$Digit4[i] <- 0
    test$Digit5[i] <- 0
    test$Digit6[i] <- 0
    test$Digit7[i] <- 0
    test$Digit8[i] <- 0
    test$Digit9[i] <- 0
  }
  else if (test$Numbers[i] == 2){
    test$Digit0[i] <- 0
    test$Digit1[i] <- 0
    test$Digit2[i] <- 1
    test$Digit3[i] <- 0
    test$Digit4[i] <- 0
    test$Digit5[i] <- 0
    test$Digit6[i] <- 0
    test$Digit7[i] <- 0
    test$Digit8[i] <- 0
    test$Digit9[i] <- 0
  }
  else if (test$Numbers[i] == 3){
    test$Digit0[i] <- 0
    test$Digit1[i] <- 0
    test$Digit2[i] <- 0
    test$Digit3[i] <- 1
    test$Digit4[i] <- 0
    test$Digit5[i] <- 0
    test$Digit6[i] <- 0
    test$Digit7[i] <- 0
    test$Digit8[i] <- 0
    test$Digit9[i] <- 0
  }
  else if (test$Numbers[i] == 4){
    test$Digit0[i] <- 0
    test$Digit1[i] <- 0
    test$Digit2[i] <- 0
    test$Digit3[i] <- 0
    test$Digit4[i] <- 1
    test$Digit5[i] <- 0
    test$Digit6[i] <- 0
    test$Digit7[i] <- 0
    test$Digit8[i] <- 0
    test$Digit9[i] <- 0
  }
  else if (test$Numbers[i] == 5){
    test$Digit0[i] <- 0
    test$Digit1[i] <- 0
    test$Digit2[i] <- 0
    test$Digit3[i] <- 0
    test$Digit4[i] <- 0
    test$Digit5[i] <- 1
    test$Digit6[i] <- 0
    test$Digit7[i] <- 0
    test$Digit8[i] <- 0
    test$Digit9[i] <- 0
  }
  else if (test$Numbers[i] == 6){
    test$Digit0[i] <- 0
    test$Digit1[i] <- 0
    test$Digit2[i] <- 0
    test$Digit3[i] <- 0
    test$Digit4[i] <- 0
    test$Digit5[i] <- 0
    test$Digit6[i] <- 1
    test$Digit7[i] <- 0
    test$Digit8[i] <- 0
    test$Digit9[i] <- 0
  }
  else if (test$Numbers[i] == 7){
    test$Digit0[i] <- 0
    test$Digit1[i] <- 0
    test$Digit2[i] <- 0
    test$Digit3[i] <- 0
    test$Digit4[i] <- 0
    test$Digit5[i] <- 0
    test$Digit6[i] <- 0
    test$Digit7[i] <- 1
    test$Digit8[i] <- 0
    test$Digit9[i] <- 0
  }
  else if (test$Numbers[i] == 8){
    test$Digit0[i] <- 0
    test$Digit1[i] <- 0
    test$Digit2[i] <- 0
    test$Digit3[i] <- 0
    test$Digit4[i] <- 0
    test$Digit5[i] <- 0
    test$Digit6[i] <- 0
    test$Digit7[i] <- 0
    test$Digit8[i] <- 1
    test$Digit9[i] <- 0
  }
  else {
    test$Digit0[i] <- 0
    test$Digit1[i] <- 0
    test$Digit2[i] <- 0
    test$Digit3[i] <- 0
    test$Digit4[i] <- 0
    test$Digit5[i] <- 0
    test$Digit6[i] <- 0
    test$Digit7[i] <- 0
    test$Digit8[i] <- 0
    test$Digit9[i] <- 1
  }
}


# comparing neuralnet to nnet packages
# first use nnet package
# y is your response, x your explanatory, size is number of neurons
# in the hidden layers, entropy = T - CE, make sure MaxWts is high
# enough for your neural net.  It might take a while...
# here our y has the 10 nodes that hold binary info on what number
# it is


########### ######################
########### 
############
###########
library(dplyr)
train_small <- sample_frac(train, .1, replace = FALSE)
test_small <- sample_frac(test, .1, replace = FALSE)

# grid <- expand.grid(.decay = c(.5, .1), .size = 28)
# fit <- train(y = train_small[,1]
  #           , x = train_small[,2:785], 
   #          data = train_small, method = "nnet", 
    #         maxit = 1000, tuneGrid = grid, trace = F, linout = 1)


# sample proportion of the train and test tdata
nn <- nnet(y = train_small[,786:795], x = train_small[,2:785], data = train_small, 
          size = 28, entropy = TRUE, MaxNWts = 24000)



predictions.nnet <- predict(nn, newdata = test_small)
head(predictions.nnet, n = 20)

colnames(predictions.nnet) <- c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

predictednumber <- colnames(predictions.nnet)[apply(
    predictions.nnet, 1, which.max)]
predictednumber <- as.numeric(predictednumber)
predictions.nnet <- cbind(predictions.nnet, predictednumber)

predictionsvsoutput <- cbind(test_small$Numbers, predictions.nnet)
head(predictionsvsoutput, n = 20)

colnames(predictionsvsoutput) <- c("actualnumber", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "predictednumber")

predictionsvsoutput <- as.data.frame(predictionsvsoutput)

for (i in 1:length(predictionsvsoutput)){
  if (predictionsvsoutput$actualnumber[i] == predictionsvsoutput$predictednumber[i]){
    predictionsvsoutput$correctestimate[i] <- 1
  } else {
    predictionsvsoutput$correctestimate[i] <- 0
  }
}

sum(predictionsvsoutput$correctestimate/length(predictionsvsoutput))


# whole data set
nn <- nnet(y = train[,786:795], x = train[,2:785], data = train, 
           size = 28, entropy = TRUE, MaxNWts = 24000)



predictions.nnet <- predict(nn, newdata = test)
head(predictions.nnet, n = 6)

colnames(predictions.nnet) <- c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

predictednumber <- colnames(predictions.nnet)[apply(
  predictions.nnet, 1, which.max)]
predictednumber <- as.numeric(predictednumber)
predictions.nnet <- cbind(predictions.nnet, predictednumber)

predictionsvsoutput <- cbind(test$Numbers, predictions.nnet)
head(predictionsvsoutput, n = 10)

colnames(predictionsvsoutput) <- c("actualnumber", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "predictednumber")

predictionsvsoutput <- as.data.frame(predictionsvsoutput)

for (i in 1:length(predictionsvsoutput)){
  if (predictionsvsoutput$actualnumber[i] == predictionsvsoutput$predictednumber[i]){
    predictionsvsoutput$correctestimate[i] <- 1
  } else {
    predictionsvsoutput$correctestimate[i] <- 0
  }
}

sum(predictionsvsoutput$correctestimate)/nrow(predictionsvsoutput)

# damn that's a high classification accuracy

head(nn$wts, n = 10)
head(nn$entropy)
head(nn$residuals, n = 10)
head(nn$fitted.values)
nn$convergence # maximum convergence was not hit if it's 0
# could up the ante for more iterations to boost the accuracy even more
# 

predictionsvsoutput[1:20,] # 3, 6, 8, 9

test$Digit9 <- NULL
test$Digit8 <- NULL
test$Digit7 <- NULL
test$Digit6 <- NULL
test$Digit5 <- NULL
test$Digit4 <- NULL
test$Digit3 <- NULL
test$Digit2 <- NULL
test$Digit1 <- NULL
test$Digit0 <- NULL

# before using display digit you need to get rid of the "digit#" columns
displayDigit(test[3,-1])
predictionsvsoutput$actualnumber[3]
predictionsvsoutput$predictednumber[3]
displayDigit(test[6,-1])
predictionsvsoutput$actualnumber[6]
predictionsvsoutput$predictednumber[6]
displayDigit(test[9,-1])
predictionsvsoutput$actualnumber[9]
predictionsvsoutput$predictednumber[9]
displayDigit(test[10,-1])
predictionsvsoutput$actualnumber[10]
predictionsvsoutput$predictednumber[10]


##
# some people are obviously really bad at writing numbers
##

# ROC curve
# need to use the prediction() function first ##!!!!@@##
prediction()
ROC.perf <- performance(predictions.nnet , measure = predictionsvsoutput$actualnumber)
plot(ROC, main = "hivpr - ROC Curves", col = "blue")

































##
# next mlp, neuralnet, monmlp## use ROCR for accuracy checking

model <- mlp(x = train[,2:785], y = train[,786:795], data = train,
           maxit = 30, initFunc = "Randomize_weights", initFuncParams = c(-.3, .3),
           learnFunc = "Std_Backpropagation", learnFuncParams = c(.2, 0),
            inputsTest = test[,2:785],
           targetsTest = test[,786:795]
)

summary(model)
weightMatrix(model)
extractNetInfo(model)

predictions <- predict(model, test[,2:785])
plotROC(predictions, test[,786:795])
plotROC(model$fitted.values, train[,786:795])

for (i in 1:length(predictions)){
  if (predictions[i] >= .5){
    predictions.thresh <- 1
  } else {
    predictions.thresh <- 0
  }
}
as.double(predictions.thresh)

confusionMatrix(as.double(predictions.thresh), as.double(test[,786:795]))

# sandbox subsets

train_small <- sample_frac(train, .1, replace = FALSE)
test_small <- sample_frac(test, .1, replace = FALSE)

?neuralnet

train_small$Numbers <- NULL
test_small$Numbers <- NULL

colnames(train_small)
y <- "Digit0 + Digit1 + Digit2 + Digit3 + Digit4 + Digit5 + Digit6 + Digit7 + Digit8 + Digit9 ~"
x <- paste(" ", colnames(train_small)
                      [-match(c("Digit0", "Digit1", "Digit2", "Digit3", "Digit4" , "Digit5" , "Digit6" , "Digit7" , "Digit8" , "Digit9"),
                              colnames(train_small))], collapse = "+")
x <- paste('Digit0 + Digit1 + Digit2 + Digit3 + Digit4 + Digit5 + Digit6 + Digit7 + Digit8 + Digit9 ~',
           paste(paste(train_small[,1:784],sep=''), collapse='+'), sep='')

formula <- as.formula(c(y,x))
formula
# for some reason the neuralnet package won't let you use ~ . for explanatory variables similar to how you would for glm()
# so above is a round about way of me specifying the 784 greyscale pixels as input nodes
# :-( on kaggle and stack overflow there are some mad angry people

x
dim(x)
colnames(train_small)
nnet2 <- neuralnet(formula = x, 
                     
                       data = train_small, hidden = 28, threshold = .01, 
                   algorithm = "rprop+",
                   err.fct = "ce", linear.output = FALSE, act.fct = "logistic")
