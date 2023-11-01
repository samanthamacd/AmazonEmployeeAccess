### FINAL AMAZON SUBMISSION ### 

# highest kaggle score has been with the smote random forest 

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
  step_smote(all_outcomes(), neighbors = 9)

prep <- prep(balanced_recipe) 
baked <- bake(prep, new_data = amazonTrain)  


my_mod2 <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification") 

forest_wf <- workflow() %>% 
  add_recipe(balanced_recipe) %>% 
  add_model(my_mod2) 

tuning_grid_forest <- grid_regular(min_n(), 
                                   mtry(range=c(1,6)),
                                   levels=10) 

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

final_preds_forest <- final_forest_wf %>% 
  predict(new_data=amazonTest, type = "prob") %>% 
  bind_cols(. , amazonTest) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(final_preds_forest, "FinalPredictions2.csv", delim=',')
