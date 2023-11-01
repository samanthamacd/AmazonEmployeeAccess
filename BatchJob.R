library(tidyverse) 
library(tidymodels)
library(vroom)
library(embed)
library(doParallel) 

num_cores <- 3
parallel::detectCores() #How many cores do I have?
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)


amazonTrain <- vroom("train.csv") 
amazonTest <- vroom("test.csv")

my_recipe <- recipe(ACTION~., data = amazonTrain) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars('ACTION'))  

amazonTrain$ACTION <- as.factor(amazonTrain$ACTION)

prep <- prep(my_recipe) 
baked <- bake(prep, new_data = amazonTrain)

log_reg_model <- logistic_reg() %>% 
  set_engine("glm") 

amazon_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(log_reg_model) %>% 
  fit(data = amazonTrain) 

amazon_predictions <- predict(amazon_wf, 
                              new_data=amazonTest, 
                              type = "prob") %>% 
  bind_cols(. , amazonTest) 

names(amazon_predictions)[1] <- "ACTION"

new_predictions <- amazon_predictions %>% 
  select(c("id", "ACTION"))

vroom_write(new_predictions, "AmazonPredictions.csv", delim=',')

# 11 October - Penalized LogReg 

preg_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>% 
  set_engine("glmnet") 

preg_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(preg_mod) %>% 
  fit(data = amazonTrain) 

tuning_grid <- grid_regular(penalty(), 
                            mixture(), levels = 10) 
folds <- vfold_cv(amazonTrain, v = 5, repeats = 1) 

cv_results <- preg_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))  

bestTune <- cv_results %>% select_best("roc_auc") 

final_wf <- preg_wf %>% finalize_workflow(bestTune) %>% 
  fit(data=amazonTrain)

final_preds <- final_wf %>% 
  predict(new_data=amazonTest, type = "prob") %>% 
  bind_cols(. , amazonTest) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(final_preds, "PregPredictions.csv", delim=',')

stopCluster(cl)
