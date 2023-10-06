# Amazon Kaggle Comp 

library(tidyverse) 
library(vroom)
library(embed)

amazonTrain <- vroom("train.csv") 
amazonTest <- vroom("test.csv")

my_recipe <- recipe(ACTION~., data = amazonTrain) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors()) 


prep <- prep(my_recipe) 
baked <- bake(prep, new_data = amazonTrain)

ncol(amazonTest)
ncol(amazonTrain)
