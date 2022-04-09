##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(recommenderlab)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#quick preview of the data
head(edx)

#the edx dataset dimensions
dim(edx)

#the edx dataset feature names
names(edx)

#Title and year are included in the same column
#would like to have the year separate to make it tidy and easier to use
#abstracting the year by adding a column of year movie was released
edx <- edx %>% 
  mutate(released = str_extract(title,"(\\d{4})"))

validation <- validation %>%
  mutate(released = str_extract(title,"(\\d{4})"))

#the timestamp feature is in a format that is not easy to read
#converting timestamp into dates
edx$timestamp <- as.POSIXct(edx$timestamp, origin="1970-01-01")

validation$timestamp <- as.POSIXct(validation$timestamp, origin="1970-01-01")

#our new timestamp gives us a readable date
#would like to have the month abstracted in order to further analyze
#adding a column with month of the rating
edx <- edx %>%
  mutate(month = months.Date(timestamp))

validation <- validation %>%
  mutate(month = months.Date(timestamp))

#How many zeros were given as ratings in the edx dataset?
sum(edx$rating == 0)

#How many threes were given as ratings in the edx dataset?
sum(edx$rating == 3)

#What percentage of ratings were given a perfect score?
mean(edx$rating == 5)

#How many different movies are in the edx dataset?
n_distinct(edx$movieId)

#How many different users are in the edx dataset?
n_distinct(edx$userId)

#Which movie has the greatest number of ratings?
edx %>% 
  group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

#What are the most given ratings in order from most to least?
edx %>% 
  group_by(rating) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

#table of distribution of movies by release year
edx %>% 
  group_by(released) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

#table of distribution of movies by month rated
edx %>%
  group_by(month) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

#In general, half star ratings are less common than whole star ratings
edx %>%
  group_by(rating) %>%
  ggplot(aes(rating)) +
  geom_bar() +
  ggtitle("Distribution of Ratings") +
  theme_minimal()

#bar graph of movies by movieId and how many times they have been rated
edx %>%
  ggplot(aes(movieId)) +
  geom_bar() +
  ggtitle("Number of times each movie has been rated") +
  theme_minimal()

#bar graph of users by userId and how many times they have rated movies
edx %>%
  ggplot(aes(userId)) +
  geom_bar() +
  ggtitle("Number of times each user has rated a movie") +
  theme_minimal()

#Analysis

#First we split the edx dataset into training and testing sets
set.seed(1, sample.kind="Rounding")
edx_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_training <- edx[-edx_index,]
edx_test <- edx[edx_index,]

# Make sure userId and movieId in validation set are also in edx_training set
#so that we do not generate N/As
temp <- validation

temp2 <- temp %>%
  semi_join(edx_training, by = 'movieId') %>%
  semi_join(edx_training, by = 'userId')

# Add rows removed from temp set back into edx_training set
removed <- anti_join(temp, temp2)

edx_training <- rbind(edx_training, removed)

rm(temp,temp2,removed)

# Make sure userId and movieId in edx_test set are also in edx_training set
#so that we do not generate N/As
temp <- edx_test

temp2 <- temp %>%
  semi_join(edx_training, by = 'movieId') %>%
  semi_join(edx_training, by = 'userId')

# Add rows removed from temp set back into edx_training set
removed <- anti_join(temp, temp2)

edx_training <- rbind(edx_training, removed)

rm(temp,temp2,removed)

#checking the dimensions of the training set and the test set to make sure they
#split 80/20
dim(edx_training)
dim(edx_test)

#check to see if there are any N/A values in the rating column
sum(is.na(edx_training$rating))
sum(is.na(edx_test$rating))

#residual mean squared error (RMSE) is what we will be using on the test set and 
#the validation set to test our predictions accuracy 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Our first model

#We are going to take the average rating of the training set and use that
#as a base line to compare other models to
mu_hat <- mean(edx_training$rating)

#Here we apply the rating average to the test set to get our error lost
naive_rmse <- RMSE(edx_test$rating, mu_hat)

#creating a data frame to store the method and results 
rmse_results <- data.frame(method = "average rating", RMSE = naive_rmse)
rmse_results

#Our second model

#Movie effect

#Here we will be generating the least squared estimate by obtaining the
#average of the rating minus the average rating for each movie
mu <- mean(edx_training$rating) 
movie_avgs <- edx_training %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#adding the movie effect to the rating average for each movie to generate our
#predictions for each movie
movie_effect <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
            
#movie effect RMSE
movie_effect_rmse <- RMSE(edx_test$rating,movie_effect)

#adding movie effect RMSE to the results table
rmse_results <- rbind(rmse_results,
                       c(method = "movie effect", 
                         RMSE = movie_effect_rmse))

#viewing the results table
rmse_results

#third model

#movie effect and user effect

#Here we will be generating the least squared estimate by obtaining the
#average of the rating minus the average rating minus the movie effect 
#for each movie
user_avgs <- edx_training %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#adding the movie effect and user effect to the rating average 
#for each movie to generate our predictions for each movie
movie_user_effect <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#movie and user effect RMSE
movie_user_effect_rmse <- RMSE(edx_test$rating, movie_user_effect)

##adding movie effect RMSE to the results table
rmse_results <- rbind(rmse_results,
                      c(method = "movie user effect", 
                      RMSE = movie_user_effect_rmse))

#viewing the results table
rmse_results

#fourth model

#Movie, User and Year Released effect

#Here we will be generating the least squared estimate by obtaining the
#average of the rating minus the average rating minus the movie effect 
#minus the user effect for each movie
year_avg <- edx_training %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by="userId") %>%
  group_by(released) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))

#adding the movie, user and year effect to the rating average 
#for each movie to generate our predictions for each movie
movie_user_year_effect <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avg, by='released') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)

#Movie, User and Year effect RMSE  
movie_user_year_effect_rmse <- RMSE(edx_test$rating, movie_user_year_effect)

#adding movie effect RMSE to the results table
rmse_results <- rbind(rmse_results,
                      c(method = "movie user year effect",
                      RMSE = movie_user_year_effect_rmse  ))

#viewing results table
rmse_results

#Fifth Model

#Movie, User, Year Released, Month Rated, effect

#Here we will be generating the least squared estimate by obtaining the
#average of the rating minus the average rating minus the movie effect 
#minus the user effect minus the year effect for each movie
month_avg <- edx_training %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avg, by='released') %>%
  group_by(month) %>%
  summarize(b_m = mean(rating - mu - b_i - b_u - b_y))

#adding the movie, user, year and month effect to the rating average 
#for each movie to generate our predictions for each movie
movie_user_year_month_effect <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avg, by='released') %>%
  left_join(month_avg, by='month') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_m) %>%
  pull(pred)

#movie, user, year, month effect RMSE
movie_user_year_month_effect_rmse <- 
  RMSE(edx_test$rating,movie_user_year_month_effect)

#adding movie effect RMSE to the results table
rmse_results <- rbind(rmse_results,
                      c(method = "movie user year month effect",
                        RMSE = movie_user_year_month_effect_rmse))

#viewing results table
rmse_results

#Sixth Model

#Since some movies have over a thousand ratings while other movies only have one
#rating, we will be regularizing the least squared estimate in order to penalize
#movies with few ratings

#regularized least squared estimates
lambda <- 3
mu <- mean(edx_training$rating)
movie_reg_avgs <- edx_training %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

#graph of the results of the movie effect regularized
tibble(original = movie_avgs$b_i, 
       regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

#adding the regularized movie effect to the rating average for each movie 
#to generate our predictions for each movie
movie_effect_reg <- edx_test %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

#Movie effect regularized RMSE
movie_effect_reg_rmse <- RMSE(edx_test$rating, movie_effect_reg)

#adding regularized movie effect RMSE to the results table
rmse_results <- rbind(rmse_results,
                      c(method = "movie effect reg",
                        RMSE = movie_effect_reg_rmse))

#viewing results table
rmse_results

#Seventh Model

#Movie and User Effect Regularized

#Since the lambada is a tuning parameter, we will run a function in order to
#find the lambada that gives us the lowest RMSE
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_training$rating)
  
  b_i <- edx_training %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_training %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    edx_training %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(edx_training$rating, predicted_ratings))
})

#plot showing the RMSE result for each lambda
qplot(lambdas, rmses)  
l <- lambdas[which.min(rmses)]

mu <- mean(edx_training$rating)

b_i <- edx_training %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- edx_training %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

#Movie and user regularized effect predictions
movie_user_reg_avgs <- 
  edx_test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#Movie and user regularized RMSE
movie_user_reg_rmse <- RMSE(edx_test$rating, movie_user_reg_avgs)

#adding the regularized movie, user effect RMSE to the results table
rmse_results <- rbind(rmse_results,
                      c(method = "movie user effect reg",
                        RMSE = movie_user_reg_rmse))

#viewing results table
rmse_results

#Eighth Model

#movie, user and year regularized effect model

#We will now be running the same model as the previous model but this time we
#will add the year released effect to the model
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_training$rating)
  
  b_i <- edx_training %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_training %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
  b_y <- edx_training %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(released) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  
  predicted_ratings <- 
    edx_training %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "released") %>%
    mutate(pred = mu + b_i + b_u + b_y) %>%
    pull(pred)
  
  return(RMSE(edx_training$rating, predicted_ratings))
})

#plot showing the RMSE result for each lambda
qplot(lambdas, rmses)  
l <- lambdas[which.min(rmses)]

mu <- mean(edx_training$rating)

b_i <- edx_training %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- edx_training %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+l))

b_y <- edx_training %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by="userId") %>%
  group_by(released) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))

#movie, user, year regularized predictions
movie_user_year_reg_avgs <- 
  edx_test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by = "released") %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)

#movie, user, year regularized RMSE
movie_user_year_reg_rmse <- RMSE(edx_test$rating, movie_user_year_reg_avgs)

#adding the regularized movie, user, year RMSE result to the result table
rmse_results <- rbind(rmse_results,
                      c(method = "movie user year reg",
                        RMSE = movie_user_year_reg_rmse))

#viewing the results table
rmse_results

#Ninth Model

#We will now be running the same model as the previous model but this time we
#will add the month effect to the model
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_training$rating)
  
  b_i <- edx_training %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_training %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
  b_y <- edx_training %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(released) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  
  b_m <- edx_training %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_y, by='released') %>%
    group_by(month) %>%
    summarize(b_m = sum(rating - mu - b_i - b_u - b_y)/(n()+1))
  
  predicted_ratings <- 
    edx_training %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "released") %>%
    left_join(b_m, by = 'month') %>%
    mutate(pred = mu + b_i + b_u + b_y + b_m) %>%
    pull(pred)
  
  return(RMSE(edx_training$rating, predicted_ratings))
})

#plot showing the RMSE result for each lambda
qplot(lambdas, rmses)  
l <- lambdas[which.min(rmses)]

mu <- mean(edx_training$rating)

b_i <- edx_training %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- edx_training %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+l))

b_y <- edx_training %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by="userId") %>%
  group_by(released) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))

b_m <- edx_training %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by = "released") %>%
  group_by(month) %>%
  summarize(b_m = sum(rating - mu - b_i - b_u - b_y)/(n()+l))
  

#movie, user, year, month regularized predictions
movie_user_year_month_reg_avgs <- 
  edx_test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by = "released") %>%
  left_join(b_m, by = "month") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_m) %>%
  pull(pred)

#movie, user, year, month regularized RMSE
movie_user_year_month_reg_rmse <- 
  RMSE(edx_test$rating, movie_user_year_month_reg_avgs)

#adding the regularized movie, user, year, month RMSE result to the result table
rmse_results <- rbind(rmse_results,
                      c(method = "movie user year month reg",
                        RMSE = movie_user_year_month_reg_rmse))

#viewing RMSE result
rmse_results

#now that we have trained and tested our models. Now we will test it against the
#validation data set and view the results
validation_pred <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by = "released") %>%
  left_join(b_m, by = "month") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_m) %>%
  pull(pred)

#validation RMSE
validation_rmse <- RMSE(validation$rating, validation_pred)

#adding the validation RMSE to the results table
rmse_results <- rbind(rmse_results, 
                      c(method = "validation",
                        RMSE = validation_rmse))
#Final results
rmse_results