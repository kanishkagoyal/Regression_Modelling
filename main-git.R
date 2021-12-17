
#===============================================================================

# Name - Kanishka Goyal
# Algorithm comparison for Regression Modelling 

#===============================================================================

# Import Libraries 
if (!("rpart.plot" %in% rownames(installed.packages()))){
  install.packages("rpart.plot")
}
if (!("caret" %in% rownames(installed.packages()))){
  install.packages("caret")
}
if (!("rpart" %in% rownames(installed.packages()))){
  install.packages("rpart")
}
if (!("glmnet" %in% rownames(installed.packages()))){
  install.packages("glmnet")
}
if (!("modelr" %in% rownames(installed.packages()))){
  install.packages("modelr")
}


library(rpart)
library(rpart.plot)
library(tidyverse)
library(stringr)
library(caret)
library(GGally)
library(modelr)
library(glmnet)

#===============================================================================

# Data Loading

Bike_Data <- read_csv("SeoulBikeData.csv")

# Data Pre processing 

# Change Column names 

# print colnames 
colnames(Bike_Data)

# Formatting Col names 
for(i in 1:length(colnames(Bike_Data))){
  colnames(Bike_Data)[i] <-  gsub("\\s*\\([^\\)]+\\)","",colnames(Bike_Data)[i])
  colnames(Bike_Data)[i] <- str_replace_all(colnames(Bike_Data)[i], " ", "_")
}

colnames(Bike_Data)

# Check for NA values 
colSums(is.na(Bike_Data))

i = 0
for(x in colSums(is.na(Bike_Data))){
  if(x != 0){
    i = i +1
  }
}

if(i==0){
  print("There are no NA values in the dataset.")
}
# No NA values found 

# Change some categorical variables to numeric 

unique(Bike_Data$Functioning_Day)
unique(Bike_Data$Holiday)

Bike_Data$Functioning_Day[Bike_Data$Functioning_Day == "Yes"] <- "1"
Bike_Data$Functioning_Day[Bike_Data$Functioning_Day == "No"] <- "0"
Bike_Data$Functioning_Day <- as.numeric(Bike_Data$Functioning_Day)

Bike_Data$Holiday[Bike_Data$Holiday == "Holiday"] <- "1"
Bike_Data$Holiday[Bike_Data$Holiday == "No Holiday"] <- "0"
Bike_Data$Holiday <- as.numeric(Bike_Data$Holiday)

# EDA to find out desired explanatory variables 

# Relation between count and seasons

Bike_Data %>%
  ggplot(aes(x = Seasons, y = Rented_Bike_Count, fill = Seasons)) +
  geom_boxplot() +
  stat_summary(fun = mean,
               geom = "point",
               shape = 20,
               size = 5,
               color = "white",
               fill ="white") +
  labs(
    x = "Seasons" ,
    y = "Bike Count every hour" ,
    title = "Bike demands w.r.t. seasons")+
  theme_minimal()+
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title.x = element_text(size = 12, hjust = 0.5),
    axis.title.y = element_text(size = 12, hjust = 0.5),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank(),
    strip.text = element_text(size = 14))

# Plot response variable 
dev.off()
Bike_Data %>%
  ggplot(aes(x =  Rented_Bike_Count, fill = ..count..)) +
  geom_histogram(binwidth = 60, color = "black") +
  labs(
    x = "Bike demand per hour" ,
    y = "Frequency" ,
    title = "Bike Demand Frequency")+
  theme_minimal()+
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title.x = element_text(size = 12, hjust = 0.5),
    axis.title.y = element_text(size = 12, hjust = 0.5),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank())

# The graph is skewed as there are few days with exceptional number of demand.
# To solve this problem, in the later stages log(response variable) can be taken
# while modelling. 

# Plotting correlation 
numerical_Bikedata <- select_if(Bike_Data, is.numeric)

numerical_Bikedata %>%
  ggcorr( name = "Rho" , low = "lightpink", high = "navy") +
  ggtitle("Correlation plot") +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5))

# There is a possibility of multicolineairty as there is some good co-relation
# between varibles.

# Removing Outliers 

#find Q1, Q3, and interquartile range for values in response column

Q1 <- quantile(Bike_Data$Rented_Bike_Count, .25)
Q3 <- quantile(Bike_Data$Rented_Bike_Count, .75)
IQR <- IQR(Bike_Data$Rented_Bike_Count)

#only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3

Bike_Data <- Bike_Data %>%
  filter(
    Rented_Bike_Count> (Q1 - 1.5*IQR),
    Rented_Bike_Count< (Q3 + 1.5*IQR)
  )

#===============================================================================

# Data Modelling 

attach(Bike_Data)

# Model 1 - (MLM)--------------------------------------------------------------

# Split the data set into test and train data 

set.seed(123)

random.samples <- Bike_Data$Rented_Bike_Count %>%
  createDataPartition(p = 0.8, list = F)

train_data <- Bike_Data[random.samples,]
test_data <- Bike_Data[-random.samples,]

# Model 1

start.time <- Sys.time()
model1 <- lm(Rented_Bike_Count ~ Hour + Temperature + Humidity + Wind_speed + 
               Solar_Radiation + Visibility + Rainfall + Snowfall  +
               Holiday + Dew_point_temperature + Seasons + Functioning_Day, 
             data=train_data)
end.time <- Sys.time()

time.taken.model1 <- end.time - start.time
time.taken.model1

summary(model1)
dev.off()
plot(model1)

# Quantitative measures
predictions_model1 <- model1 %>% predict(test_data) 

RMSE_model1 <- RMSE(predictions_model1, test_data$Rented_Bike_Count) 
Rsquare_model1 <- R2(predictions_model1, test_data$Rented_Bike_Count)
mae_model1 <- MAE(predictions_model1, test_data$Rented_Bike_Count)


# This residual plot is not a random scatter. There is a definite pattern in the
# residuals which indicates that the features are not independent of each other.

# In the normal-QQ plot, points are not following a straight line and also there
# are still many outliers. 

# Model- 2 Polynomial Model -----------------------------------------------------

# Adding log transformation to response variable 

train_data$log_count <- log(train_data$Rented_Bike_Count)
test_data$log_count <- log(test_data$Rented_Bike_Count)

# Changing log values where it became NaN due to 0 inputs 

train_data$log_count[train_data$log_count < 0] <- 0
test_data$log_count[test_data$log_count < 0] <- 0

start.time <- Sys.time()
model2 <- lm(log_count ~ Hour + Temperature + Dew_point_temperature  + 
               Functioning_Day + Solar_Radiation +  Rainfall + Humidity +
               Holiday + Seasons + Humidity:Dew_point_temperature +
               Rainfall:Humidity, 
             data=train_data)
end.time <- Sys.time()

time.taken.model2 <- end.time - start.time
time.taken.model2
summary(model2)

# All kinds of residual plots 
dev.off()
plot(model2)

predictions_model2 <- model2 %>% predict(test_data) 

# Quantitaive measures 
RMSE_model2 <- RMSE(predictions_model2, test_data$log_count) 
Rsquare_model2 <- R2(predictions_model2, test_data$log_count)
mae_model2 <- MAE(predictions_model2, test_data$log_count)
# 3rd Algorithm Lasso ----------------------------------------------------------

# Due to high multi-collineairty between explanatory variables, a regularized
# model is needed.

train_selective <- train_data[, 3:15]
test_selective <- test_data[, 3:15]

x <- model.matrix(log_count~., train_selective)[, -1]
x.test <- model.matrix(log_count~., test_selective)[, -1]

y <- train_selective$log_count
y.test <- test_selective$log_count  

set.seed(123)

start.time <- Sys.time()

sel_lambda_lasso <- cv.glmnet(x, y, alpha = 1)

lassoModel <- glmnet(x, y, alpha = 1, lambda = sel_lambda_lasso$lambda.min)
end.time <- Sys.time()
time.taken.model3 <- end.time - start.time
time.taken.model3

predictions_model3 <- lassoModel %>% predict(x.test) %>% as.vector()

# Quantitative measures
RMSE_model3 = RMSE(predictions_model3, y.test)
Rsquare_model3 = R2(predictions_model3, y.test)
mae_model3 = MAE(predictions_model3, test_selective$log_count)

lassoModel
RMSE_model3
Rsquare_model3
mae_model3

# 4rth Algorithm KNN ------------------------------------------------------------

# While performing KNN, standardizing the scale is important.

start.time <- Sys.time()
knnregression <- train(log_count ~ Hour + Temperature + Humidity + Wind_speed + 
                         Solar_Radiation + Visibility + Rainfall + Snowfall  +
                         Holiday + Dew_point_temperature + Seasons + Functioning_Day, 
                       data = train_data,
                       method = "knn",
                      preProcess = c("center","scale"),
                       trControl = trainControl("cv", number = 20), 
                       tuneLength = 10
)
end.time <- Sys.time()
time.taken.model4 <- end.time - start.time
time.taken.model4

# Print summarised reults

knnregression

# Plot for trying different values of k across RMSE

dev.off()
plot(knnregression)

# Quantitative measures 

predictions_model4 <- predict(knnregression, test_data)
RMSE_model4 <- RMSE(predictions_model4, test_data$log_count)
Rsquare_model4 <- R2(predictions_model4, test_data$log_count)
mae_model4 <- MAE(predictions_model4, test_data$log_count)


# 5th Algorithm Poison --------------------------------------------------------

start.time <- Sys.time()

poisson_model <- glm(log_count~ Hour + Temperature + Humidity + Wind_speed + 
                       Solar_Radiation + Visibility + Rainfall + Snowfall  +
                       Holiday + Dew_point_temperature + Seasons + Functioning_Day,
                     data = train_data)
                 #    family = poisson)

end.time <- Sys.time()
time.taken.model5 <- end.time - start.time
time.taken.model5


summary(poisson_model)

# predictions
predictions_model5 <- predict(poisson_model, test_data)

# calculating evaluation metrics
RMSE_model5 = RMSE(predictions_model5, test_data$log_count)
Rsquare_model5 = R2(predictions_model5, test_data$log_count)
mae_model5 = MAE(predictions_model5, test_data$log_count)

# 4 plots
par(mfrow = c(2,2)) 
plot(poisson_model, which = c(1:4))
#===============================================================================


#Evaluation

eval_df <- data.frame(
  Name = c("MLM","Polynomial", "Lasso", "KNN", "Poisson"),
  Execution_time = c(time.taken.model1, time.taken.model2, time.taken.model3,
                     time.taken.model4,time.taken.model5),
  RMSE = c(RMSE_model1, RMSE_model2, RMSE_model3,RMSE_model4,RMSE_model5),
  Rsquare = c(Rsquare_model1, Rsquare_model2, Rsquare_model3, Rsquare_model4,
              Rsquare_model5),
  MAE = c(mae_model1, mae_model2, mae_model3, mae_model4, mae_model5)
)

# Wide to Long

eval_df_long <- gather(eval_df, Type , Value, Execution_time: MAE, factor_key=TRUE)

# Plotting
dev.off()

eval_df_long %>%
  filter(Type != "Execution_time", Name != "MLM") %>%
  ggplot(aes(x = Name, y = Value, fill = Name)) +
  geom_bar(stat= "identity") +
  facet_wrap(~Type) +
  labs(
    x = "Algorithm name" ,
    y = "Value" ,
    title = "Quantitative metrics for 4 models")+
  theme_minimal()+
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title.x = element_text(size = 12, hjust = 0.5),
    axis.title.y = element_text(size = 12, hjust = 0.5),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank(),
    strip.text = element_text(size = 14))

#2 Another for time 

eval_df_long %>%
  filter(Type == "Execution_time") %>%
  ggplot(aes(x = Name, y = Value, fill = Name)) +
  geom_point() +
  labs(
    x = "Algorithm name" ,
    y = "Value" ,
    title = "Quantitative metrics for 4 models")+
  theme_minimal()+
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title.x = element_text(size = 12, hjust = 0.5),
    axis.title.y = element_text(size = 12, hjust = 0.5),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank(),
    strip.text = element_text(size = 14))





