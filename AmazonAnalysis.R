library(vroom)
library(patchwork)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)

# set working directory
setwd("C:/Users/davis/OneDrive - Brigham Young University/Documents/skool/new/stat 348/AmazonEmployeeAccess/AmazonEmployeeAccess")

# read in training data
train <- vroom("train.csv")
test <- vroom("test.csv")

train$ACTION <- as.factor(train$ACTION)

# 2 plots for EDA - column characterizations and bar plot of response (ACTION)
(DataExplorer::plot_intro(train)) / (DataExplorer::plot_bar(train))

# make recipe, prep for baking, then bake
recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>% # all uncommon categories for all variables converted to other
  step_dummy(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

prep <- prep(recipe)

baked <- bake(prep, new_data = NULL)


## logistic regression

log_model <- logistic_reg() %>%
  set_engine("glm")

log_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(log_model) %>%
  fit(data = train)

log_prediction <- predict(log_workflow,
                          new_data = test,
                          type = "prob")

log_submission <- data.frame(id = test$id, 
                             ACTION = log_prediction$.pred_1)

write.csv(log_submission, "log_submission.csv", row.names = F)


# penalized logistic regression

pen_log_model <- logistic_reg(mixture = tune(),
                              penalty = tune()) %>%
  set_engine("glmnet")

pen_log_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(pen_log_model)

pen_log_tuning_grid <- grid_regular(penalty(),
                                    mixture(),
                                    levels = 5)

pen_log_folds <- vfold_cv(train, v = 5, repeats = 1)

pen_cv_results <- pen_log_workflow %>%
  tune_grid(resamples = pen_log_folds,
            grid = pen_log_tuning_grid,
            metrics = metric_set(roc_auc))

pen_log_besttune <- pen_cv_results %>%
  select_best("roc_auc")

final_pen_log_workflow <- pen_log_workflow %>%
  finalize_workflow(pen_log_besttune) %>%
  fit(data = train)

pen_log_predictions <- final_pen_log_workflow %>%
  predict(new_data = test, type = "prob")

pen_log_submission <- data.frame(id = test$id, ACTION = pen_log_predictions$.pred_1)

write.csv(pen_log_submission, "pen_log_submission.csv", row.names = F)


# classification random forest

class_rf_model <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

class_rf_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(class_rf_model)

class_rf_tuning_grid <- grid_regular(mtry(range = c(1, 10)),
                                     min_n(),
                                     levels = 5)

class_rf_folds <- vfold_cv(train, v = 5, repeats = 1)

class_rf_results <- class_rf_wf %>%
  tune_grid(resamples = class_rf_folds,
            grid = class_rf_tuning_grid)

best_class_rf <- class_rf_results %>%
  select_best("roc_auc")

final_class_rf_wf <- class_rf_wf %>%
  finalize_workflow(best_class_rf) %>%
  fit(data = train)

class_rf_predictions <- final_class_rf_wf %>%
  predict(new_data = test)

class_rf_submission <- data.frame(id = test$id, ACTION = class_rf_predictions$.pred_class)

write.csv(class_rf_submission, file = "class_rf_submission.csv", row.names = F)


# Naive Bayes

recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>% # all uncommon categories for all variables converted to other
  # step_dummy(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(nb_model)

nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 3)

nb_folds <- vfold_cv(train, v = 5, repeats = 1)

nb_results <- nb_wf %>%
  tune_grid(resamples = nb_folds,
            grid = nb_tuning_grid)

best_nb_wf <- nb_results %>%
  select_best("roc_auc")

final_nb_wf <- nb_wf %>%
  finalize_workflow(best_nb_wf) %>%
  fit(data = train)

nb_prediction <- final_nb_wf %>%
  predict(new_data = test, type = "prob")

nb_submission <- data.frame(id = test$id, ACTION = nb_prediction$.pred_1)

write.csv(nb_submission, "nb_submission.csv", row.names = F)
