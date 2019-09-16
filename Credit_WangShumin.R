library(neuralnet)
library(tictoc)
credit <- read.csv("NN_workshop_bankloan.csv")
head(credit)
str(credit)
colnames(credit)
summary(credit)
table(credit$CreditScore)

credit_sim <- subset(credit, select = -c(CustomerID))

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
credit_norm <- as.data.frame(lapply(credit_sim, normalize))

set.seed(123)
splitdata <- sample.split(credit_norm$CreditScore, SplitRatio = 0.8)
train_set <- credit_norm[splitdata,]
test_set <- credit_norm[!splitdata,]
nrow(train_set)/nrow(credit_norm)
nrow(test_set)/nrow(credit_norm)

param_nodes_hidden_layer <- c(2,3,1) 
param_max_iteration <- 1e6 
param_learning_rate <- 0.1 

names <- colnames(credit_norm)
f <- as.formula(paste("CreditScore ~",paste(names[!names %in% "CreditScore"], collapse = "+ ")))
tic("Neural network training")
nnmodel <- neuralnet(f, data = train_set, 
                     hidden=param_nodes_hidden_layer, 
                     stepmax=param_max_iteration, 
                     learningrate = param_learning_rate,
                     linear.output=FALSE)
toc()
plot(nnmodel)

mypredict <- compute(nnmodel, test_set[,-1])$net.result
mypredict <- sapply(mypredict, round, digits=0)
results = data.frame(actual = test_set$CreditScore, prediction = mypredict)

table(results)
