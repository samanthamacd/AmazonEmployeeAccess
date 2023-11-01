# SMOTE MODELS 

library(tidyverse) 
library(tidymodels)
library(vroom)
library(embed)
library(themis)

amazonTrain <- vroom("train.csv") 
amazonTest <- vroom("test.csv")
amazonTrain$ACTION <- as.factor(amazonTrain$ACTION)

balanced_recipe <- recipe(ACTION~., data = amazonTrain) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars('ACTION')) %>% 
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .9) %>% 
  step_smote(all_outcomes(), neighbors = 6)

prep <- prep(balanced_recipe) 
baked <- bake(prep, new_data = amazonTrain)  


# LOGISTIC REGRESSION - not good very bad 

log_reg_model <- logistic_reg() %>% 
  set_engine("glm") 

amazon_wf <- workflow() %>% 
  add_recipe(balanced_recipe) %>% 
  add_model(log_reg_model) %>% 
  fit(data = amazonTrain) 

amazon_predictions <- predict(amazon_wf, 
                              new_data=amazonTest, 
                              type = "prob") %>% 
  bind_cols(. , amazonTest) 

names(amazon_predictions)[1] <- "ACTION"

new_predictions <- amazon_predictions %>% 
  select(c("id", "ACTION"))

vroom_write(new_predictions, "SMOTE_LOGREG.csv", delim=',')

# PENALIZED REGRESSION - 0.7836 

preg_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>% 
  set_engine("glmnet") 

preg_wf <- workflow() %>% 
  add_recipe(balanced_recipe) %>% 
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

final_preg_wf <- preg_wf %>% finalize_workflow(bestTune) %>% 
  fit(data=amazonTrain)

final_preds_preg <- final_preg_wf %>% 
  predict(new_data=amazonTest, type = "prob") %>% 
  bind_cols(. , amazonTest) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(final_preds_preg, "SMOTEPregPredictions.csv", delim=',')

# RANDOM FORESTS - .8414 slay!!!!

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification") 

forest_wf <- workflow() %>% 
  add_recipe(balanced_recipe) %>% 
  add_model(my_mod) 

tuning_grid_forest <- grid_regular(min_n(), 
                                   mtry(range=c(1,10))) 

folds <- vfold_cv(amazonTrain, v = 5, repeats=1)

CV_results <- forest_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid_forest, 
            metrics = metric_set(roc_auc))  

bestTune <- CV_results %>% 
  select_best("roc_auc") 

final_forest_wf <- forest_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data=amazonTrain) 

more_preds <- final_forest_wf %>% 
  predict(new_data = amazonTest)

final_preds_forest <- final_forest_wf %>% 
  predict(new_data=amazonTest, type = "prob") %>% 
  bind_cols(. , amazonTest) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(final_preds_forest, "SMOTEForestPredictions.csv", delim=',')

# NAIVE BAYES - .78308 

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(balanced_recipe) %>% 
  add_model(nb_model) 

tuning_grid_bayes <- grid_regular(smoothness(), Laplace())  

folds <- vfold_cv(amazonTrain, v = 5, repeats=1)

CV_results_nb <- nb_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid_bayes, 
            metrics = metric_set(roc_auc))  

bestTune_nb <- CV_results_nb %>% 
  select_best("roc_auc") 

final_nb_wf <- nb_wf %>% 
  finalize_workflow(bestTune_nb) %>% 
  fit(data=amazonTrain) 

nb_preds <- final_nb_wf %>% 
  predict(new_data = amazonTest)

final_preds_nb <- final_nb_wf %>% 
  predict(new_data=amazonTest, type = "prob") %>% 
  bind_cols(. , amazonTest) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(final_preds_nb, "SMOTE_NBPredictions.csv", delim=',')

# KNN - .7979 

knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn") 

knn_wf <- workflow() %>% 
  add_recipe(balanced_recipe) %>% 
  add_model(knn_model) 


tuning_grid_knn <- grid_regular(neighbors())   

folds <- vfold_cv(amazonTrain, v = 5, repeats=1)

CV_results_knn <- knn_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid_knn, 
            metrics = metric_set(roc_auc))  

bestTune_knn <- CV_results_knn %>% 
  select_best("roc_auc") 

final_knn_wf <- knn_wf %>% 
  finalize_workflow(bestTune_knn) %>% 
  fit(data=amazonTrain) 

knn_preds <- final_knn_wf %>% 
  predict(new_data = amazonTest)

final_preds_knn <- final_knn_wf %>% 
  predict(new_data=amazonTest, type = "prob") %>% 
  bind_cols(. , amazonTest) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(final_preds_knn, "SMOTE_KNNPredictions.csv", delim=',')


