library(vroom)
library(patchwork)
library(tidymodels)
library(embed)

#set working directory
setwd("C:/Users/davis/OneDrive - Brigham Young University/Documents/skool/new/stat 348/AmazonEmployeeAccess/AmazonEmployeeAccess")

# read in training data
train <- vroom("train.csv")

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
