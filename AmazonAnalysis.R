library(vroom)
library(patchwork)
library(tidymodels)
library(embed)

# set working directory
setwd("C:/Users/davis/OneDrive - Brigham Young University/Documents/skool/new/stat 348/AmazonEmployeeAccess/AmazonEmployeeAccess")

# read in training data
train <- vroom("train.csv")
test <- vroom("test.csv")

# 2 plots for EDA - column characterizations and bar plot of response (ACTION)
(DataExplorer::plot_intro(train)) / (DataExplorer::plot_bar(train))

# make recipe, prep for baking, then bake
recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>% # all uncommon categories for all variables converted to other
  step_dummy(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

train$ACTION <- as.factor(train$ACTION)

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
