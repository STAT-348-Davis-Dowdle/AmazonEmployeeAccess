library(vroom)
library(patchwork)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)
library(doParallel)

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


# KNN
recipe <- recipe(ACTION ~ ., data = train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>% # all uncommon categories for all variables converted to other
  # step_dummy(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(knn_model)

knn_tuning_grid <- grid_regular(neighbors(range = c(5, 25)),
                                levels = 5)

knn_folds <- vfold_cv(train, v = 5, repeats = 1)

knn_results <- knn_wf %>%
  tune_grid(resamples = knn_folds,
            grid = knn_tuning_grid)

knn_best <- knn_results %>%
  select_best("roc_auc")

knn_final_wf <- knn_wf %>%
  finalize_workflow(knn_best) %>%
  fit(data = train)

knn_prediction <- knn_final_wf %>%
  predict(new_data = test, type = "prob")

knn_submission <- data.frame(id = test$id, ACTION = knn_prediction$.pred_1)

write.csv(knn_submission, file = "knn_submission.csv", row.names = F)


# Principal Component Reduction
pca_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>% # all uncommon categories for all variables converted to other
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold= .9)

pca_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

pca_wf <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(pca_model)

pca_tuning_grid <- grid_regular(neighbors(),
                                levels = 5)

pca_folds <- vfold_cv(train, v = 5, repeats = 1)

pca_results <- pca_wf %>%
  tune_grid(resamples = pca_folds,
            grid = pca_tuning_grid)

pca_best <- pca_results %>%
  select_best("roc_auc")

pca_final_wf <- pca_wf %>%
  finalize_workflow(pca_best) %>%
  fit(data = train)

pca_prediction <- pca_final_wf %>%
  predict(new_data = test, type = "prob")

pca_submission <- data.frame(id = test$id, ACTION = pca_prediction$.pred_1)

write.csv(pca_submission, file = "pca_submission.csv", row.names = F)


pca_nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

pca_nb_wf <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(pca_nb_model)

pca_nb_tuning_grid <- grid_regular(Laplace(),
                                   smoothness(),
                                   levels = 5)

pca_nb_folds <- vfold_cv(train, v = 5, repeats = 1)

pca_nb_results <- pca_nb_wf %>%
  tune_grid(resamples = pca_nb_folds,
            grid = pca_nb_tuning_grid)

pca_best_nb_wf <- pca_nb_results %>%
  select_best("roc_auc")

final_pca_nb_wf <- pca_nb_wf %>%
  finalize_workflow(pca_best_nb_wf) %>%
  fit(data = train)

pca_nb_prediction <- final_pca_nb_wf %>%
  predict(new_data = test, type = "prob")

pca_nb_submission <- data.frame(id = test$id, ACTION = pca_nb_prediction$.pred_1)

write.csv(pca_nb_submission, "pca_nb_submission.csv", row.names = F)


# Support Vector Machines
svm_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>% # all uncommon categories for all variables converted to other
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

svm_model <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm_model)

svm_tuning_grid <- grid_regular(rbf_sigma(), cost(), levels = 3)

svm_folds <- vfold_cv(train, v = 5, repeats = 1)

svm_results <- svm_wf %>%
  tune_grid(resamples = svm_folds,
            grid = svm_tuning_grid)

svm_best <- svm_results %>%
  select_best("roc_auc")

svm_final_wf <- svm_wf %>%
  finalize_workflow(svm_best) %>%
  fit(data = train)

svm_predictions <- svm_final_wf %>%
  predict(new_data = test, type = "prob")

svm_submission <- data.frame(id = test$id, ACTION = svm_predictions$.pred_1)

write.csv(svm_submission, "svm_submission.csv", row.names = F)


# Imbalanced Data
bal_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_smote(all_outcomes(), neighbors = 3)

bake(prep(bal_recipe), new_data = test)


log_model <- logistic_reg() %>%
  set_engine("glm")

log_workflow <- workflow() %>%
  add_recipe(bal_recipe) %>%
  add_model(log_model) %>%
  fit(data = train)

log_prediction <- predict(log_workflow,
                          new_data = test,
                          type = "prob")

log_submission <- data.frame(id = test$id, 
                             ACTION = log_prediction$.pred_1)

write.csv(log_submission, "log_bal_submission.csv", row.names = F)


pen_log_model <- logistic_reg(mixture = tune(),
                              penalty = tune()) %>%
  set_engine("glmnet")

pen_log_workflow <- workflow() %>%
  add_recipe(bal_recipe) %>%
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

write.csv(pen_log_submission, "pen_log_bal_submission.csv", row.names = F)


class_rf_model <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

class_rf_wf <- workflow() %>%
  add_recipe(bal_recipe) %>%
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

write.csv(class_rf_submission, file = "class_rf_bal_submission.csv", row.names = F)


nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(bal_recipe) %>%
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

write.csv(nb_submission, "nb_bal_submission.csv", row.names = F)


knn_bal_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 3)

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(knn_bal_recipe) %>%
  add_model(knn_model)

knn_tuning_grid <- grid_regular(neighbors(),
                                levels = 5)

knn_folds <- vfold_cv(train, v = 5, repeats = 1)

knn_results <- knn_wf %>%
  tune_grid(resamples = knn_folds,
            grid = knn_tuning_grid)

knn_best <- knn_results %>%
  select_best("roc_auc")

knn_final_wf <- knn_wf %>%
  finalize_workflow(knn_best) %>%
  fit(data = train)

knn_prediction <- knn_final_wf %>%
  predict(new_data = test, type = "prob")

knn_submission <- data.frame(id = test$id, ACTION = knn_prediction$.pred_1)

write.csv(knn_submission, file = "knn_bal_submission.csv", row.names = F)


pca_nb_bal_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 3) %>%
  step_pca(all_predictors(), threshold= .9)

pca_nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

pca_nb_wf <- workflow() %>%
  add_recipe(pca_nb_bal_recipe) %>%
  add_model(pca_nb_model)

pca_nb_tuning_grid <- grid_regular(Laplace(),
                                   smoothness(),
                                   levels = 5)

pca_nb_folds <- vfold_cv(train, v = 5, repeats = 1)

pca_nb_results <- pca_nb_wf %>%
  tune_grid(resamples = pca_nb_folds,
            grid = pca_nb_tuning_grid)

pca_best_nb_wf <- pca_nb_results %>%
  select_best("roc_auc")

final_pca_nb_wf <- pca_nb_wf %>%
  finalize_workflow(pca_best_nb_wf) %>%
  fit(data = train)

pca_nb_prediction <- final_pca_nb_wf %>%
  predict(new_data = test, type = "prob")

pca_nb_submission <- data.frame(id = test$id, ACTION = pca_nb_prediction$.pred_1)

write.csv(pca_nb_submission, "pca_nb_bal_submission.csv", row.names = F)


svm_bal_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 3)

svm_model <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(svm_bal_recipe) %>%
  add_model(svm_model)

svm_tuning_grid <- grid_regular(rbf_sigma(), cost(), levels = 3)

svm_folds <- vfold_cv(train, v = 5, repeats = 1)

svm_results <- svm_wf %>%
  tune_grid(resamples = svm_folds,
            grid = svm_tuning_grid)

svm_best <- svm_results %>%
  select_best("roc_auc")

svm_final_wf <- svm_wf %>%
  finalize_workflow(svm_best) %>%
  fit(data = train)

svm_predictions <- svm_final_wf %>%
  predict(new_data = test, type = "prob")

svm_submission <- data.frame(id = test$id, ACTION = svm_predictions$.pred_1)

write.csv(svm_submission, "svm_bal_submission.csv", row.names = F)


# Final Submission
final_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

final_model <- rand_forest(mtry = 1,
                           min_n = 15, 
                           trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger")

final_wf <- workflow() %>%
  add_recipe(final_recipe) %>%
  add_model(final_model) %>%
  fit(data = train)

final_predictions <- final_wf %>%
  predict(new_data = test, type = "prob")

final_submission <- data.frame(id = test$id, ACTION = final_predictions$.pred_1)

write.csv(final_submission, "final_submission.csv", row.names = F)
