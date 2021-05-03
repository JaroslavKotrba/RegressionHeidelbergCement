# HeidelbergCement Material Strength

# -------------------------------------------------------------------------------------------------
# Import ------------------------------------------------------------------------------------------

setwd("C:/Users/HP/OneDrive/Documents/R/Heidelberg Cement")
dir()

library(readxl)
data <- read_excel("interview_dataset.xlsx", sheet = 1)

# Exploratory analysis

# -------------------------------------------------------------------------------------------------
# Summary -----------------------------------------------------------------------------------------

summary(data)
str(data)

# -------------------------------------------------------------------------------------------------
# NAs ---------------------------------------------------------------------------------------------

library(Amelia) #missing data
missmap(data, main = 'Missing Map', col = c('red', 'green'), rank.order = F)

# library(naniar)
# vis_miss(data)

# -------------------------------------------------------------------------------------------------
# Correlation -------------------------------------------------------------------------------------

library(tidyr)
library(corrplot)
nums <- drop_na(as.data.frame(dplyr::select_if(data, is.numeric)))
corr <- as.data.frame(cor(nums,use="pairwise.complete.obs"))
corrplot::corrplot(cor(nums))

# -------------------------------------------------------------------------------------------------
# Distribution ------------------------------------------------------------------------------------

library(purrr) # selecting a particular variable
library(tidyr)
library(ggplot2)

x <- nums$X_17 # CHANGE
x <- na.omit(x)
SD <- sd(x)
mean.x <- mean(x)
hist(x, breaks = 30, density = 20, prob=TRUE,
     main="",
     xlab="Variable",
     ylab="Density",
     cex.lab=1.2)
quant <- seq(min(x),max(x),length=100)
normaldens <- dnorm(quant,mean=mean.x,sd=SD)
lines(quant,normaldens,col="red",lwd=2)
lines(density(x), col="blue",lwd=2)
legend("topleft",c("normal","observed"),lty=c(1,1),
       col=c("red","blue"),lwd=2)

data %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram(bins=30)

# Feature engineering

# -------------------------------------------------------------------------------------------------
# Feature engineering preprocessing ---------------------------------------------------------------

library(tidyverse)
unique(data$X_26) # X_26 separate time and date
data$X_26 <- as.character(data$X_26)
data <- separate(data=data, col=X_26, into=c('date', 'time'), sep=' ')

library(ggplot2) # material strength over time
require(scales)
library(plotly)
data$date <- as.Date(data$date)
g <- ggplot(data = data,
            mapping = aes(x = date, y = y)) +
  geom_point(color="blue") +
  ggtitle("Material strength (y) through the time") +
  ylab("") +
  xlab("") +
  scale_x_date(labels = function(x) format(x, "%Y-%m-%d")) +
  theme_bw()

ggplotly(g)

unique(data$X_28) # X_28 from a string into two categories
data <- separate(data=data, col=X_28, into=c('X_28', 'X_29'), sep='---')
data$X_28 <- as.numeric(data$X_28)
data$X_29 <- as.numeric(data$X_29)

library(ggplot2) # separation of trends based on one category
require(scales)
library(plotly)
data$date <- as.Date(data$date)
g <- ggplot(data = data,
            mapping = aes(x = date, y = y)) +
  geom_point(color="blue") +
  ggtitle("Material strength (y) through the time") +
  ylab("") +
  xlab("") +
  facet_grid(facets = X_29 ~ .) +
  scale_x_date(labels = function(x) format(x, "%Y-%m-%d")) +
  theme_bw()

ggplotly(g)

unique(data$X_14) # X_14 deleting of a column with NAs only
data$X_14 <- NULL

unique(data$X_12) # X_12 simple linear regression for imputation
X_12 <- subset(data, select = c(X_11,X_12))
colnames(X_12) <- c('X', 'y')
X_12 <- subset(X_12, !(is.na(y)))

library(caTools)
set.seed(1001)
split <- sample.split(X_12$y, SplitRatio = 0.7)
train <- subset(X_12, split==T)
test <- subset(X_12, split==F)

model <- lm(y ~ X, data=train) # linear regression for getting the coefficient
summary(model)

length <- length(colnames(X_12))
y_pred <- predict(model, test[ ,-c(length)])
y_true <- test$y

coefficients <- as.data.frame(model$coefficients)
coef <- coefficients[2,]
inter <- coefficients[1,]
mean_absolute_error <- function(x,y)
{mean(abs(x-y))}

paste0("Coefficient: ", round(coef,4))
paste0("Intercept: ", round(inter,4))
paste0("Mean absolute error: ", round(mean_absolute_error(y_true, y_pred),4))

library(ggplot2) # visualisation of regression for imputation
library(plotly)
tr <- ggplot(train, aes(train$X, train$y)) +
  geom_point(color='blue', size=3) +
  geom_line(aes(train$X, predict(model, newdata=train)), color='red', size=1) +
  ggtitle("predicted X_12") +
  xlab("X_11") +
  ylab("X_12") +
  theme_bw()
ggplotly(tr)

data$X_12 <- round(ifelse(is.na(data$X_12), data$X_11*coef, data$X_12)) # imputation with help of the coefficient
length(which(!is.na(data$X_12)))

unique(data$X_0) # X_0 deleting of a column with lot of NAs
summary(data$X_0)
data$X_0 <- NULL

unique(data$X_17) # X_17 mean imputation
mean <- mean(data$X_17, na.rm = T)
mean

data$X_17 <- round(ifelse(is.na(data$X_17), mean, data$X_17),1)

# REST of NAs and outliers deleted
library(tidyr)
data <- data %>% drop_na()
data

# Models

# -------------------------------------------------------------------------------------------------
# Model1 - Multiple Linear Regression X -----------------------------------------------------------

dput(colnames(data)) # splitting data into train and test set
df <- data[c("X_1", "X_2", "X_3", "X_4", "X_5", "X_6", "X_7", "X_8", "X_9", 
             "X_10", "X_11", "X_12", "X_15", "X_16", "X_17", "X_18", 
             "X_19", "X_28", "X_29", "y")]

library(caTools)
set.seed(123)
split <- sample.split(df$y, SplitRatio = 0.8)
train <- subset(df, split==T)
test <- subset(df, split==F)

model1 <- lm(y ~ ., data=train) # multiple linear regression model
summary(model1)

length <- length(colnames(df))
y_pred <- predict(model1, test[ ,-c(length)])
y_true <- test$y
outcome <- cbind(test, y_pred)
outcome$difference <- outcome$y - outcome$y_pred

# Percentage
outcome$difference.percentage <- round(outcome$difference/(outcome$y/100),6)
paste0("Percentage difference: ", round(mean(abs(outcome$difference.percentage)),2), "%") # 2.63%

# MAE
mean_absolute_error <- function(x,y)
{mean(abs(x-y))}
paste0("'Mean absolute error: ", round(mean_absolute_error(y_true, y_pred),4)) # all 1.4511

# -------------------------------------------------------------------------------------------------
# Model2 - Multiple Linear Regression y Error -----------------------------------------------------

dput(colnames(data)) # splitting data into train and test set
df <- data[c("y_0", "y_1", "y_2", "y_3", "y_4", "y_5", "y")]

df$e_0 <- df$y_0 - df$y
df$e_1 <- df$y_1 - df$y
df$e_2 <- df$y_2 - df$y
df$e_3 <- df$y_3 - df$y
df$e_4 <- df$y_4 - df$y
df$e_5 <- df$y_5 - df$y
str(df)

df$y <- rowMeans(df[,8:13])
df <- df[, -c(8:13)]

library(caTools)
set.seed(123)
split <- sample.split(df$y, SplitRatio = 0.8)
train <- subset(df, split==T)
test <- subset(df, split==F)

model2 <- lm(y ~ ., data=train) # multiple linear regression model
summary(model2)

length <- length(colnames(df))
y_pred <- predict(model2, test[ ,-c(length)])
y_true <- test$y
outcome <- cbind(test, y_pred)
outcome$difference <- outcome$y - outcome$y_pred

# MAE
mean_absolute_error <- function(x,y)
{mean(abs(x-y))}
paste0("'Mean absolute error: ", round(mean_absolute_error(y_true, y_pred),4)) # all 0.1274

# -------------------------------------------------------------------------------------------------
# Model3 - Random Forest Regression X -------------------------------------------------------------

dput(colnames(data)) # splitting data into train and test set
df <- data[c("X_1", "X_2", "X_3", "X_4", "X_5", "X_6", "X_7", "X_8", "X_9", 
             "X_10", "X_11", "X_12", "X_15", "X_16", "X_17", "X_18", 
             "X_19", "X_28", "X_29", "y")]

library(caTools)
set.seed(123)
split <- sample.split(df$y, SplitRatio = 0.8)
train <- subset(df, split==T)
test <- subset(df, split==F)

library(randomForest)
model3 <- randomForest(x=train[1:19], # random forest regression model
                      y=train$y,
                      ntree = 500)

length <- length(colnames(df))
y_pred <- predict(model3, test[ ,-c(length)])
y_true <- test$y
outcome <- cbind(test, y_pred)
outcome$difference <- outcome$y - outcome$y_pred

# Percentage
outcome$difference.percentage <- round(outcome$difference/(outcome$y/100),6)
paste0("Percentage difference: ", round(mean(abs(outcome$difference.percentage)),2), "%") # 1.76%

# MAE
mean_absolute_error <- function(x,y)
{mean(abs(x-y))}
paste0("'Mean absolute error: ", round(mean_absolute_error(y_true, y_pred),4)) # all 0.9705

# Conclusion

### The random forest model performed the best out of the models that I tried: decision tree regression, multiple linear regression, SVM and ANN. The random forest does not take into consideration outliers and even in comparison to the linear regression model where is high linear dependency, performs very well.

### The Linear regression model of y error shows a really low mean absolute error because we try to estimate the error out of the six measurements. One can see high linear dependency, which resulted in a choice of multiple linear regression model.

               
               
