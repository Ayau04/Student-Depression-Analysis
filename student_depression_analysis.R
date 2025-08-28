install.packages(c("glmnet", "tidyverse", "caret"))
install.packages(c("pROC"))
library(glmnet)
library(tidyverse)
library(caret)
library(pROC)

# Import the dataset
data <- read_csv("student_depression_dataset.csv")

str(data)
problems(data)

# Remove ID columns and non-predictors if any
data_clean <- data %>%
  drop_na()  # remove rows with missing values

set.seed(123)  # for reproducibility
data_reduced <- data_clean %>% sample_n(450)
nrow(data_reduced)

table(data_reduced$Sleep)
# remove the 1 "others" response from diet
data_reduced <- data_reduced %>%
  filter(Diet != "Others")

# sleep to factor
data_reduced$Sleep <- as.factor(data_reduced$Sleep)
# diet to numeric
data_reduced <- data_reduced %>%
  mutate(
    Diet = case_when(
      Diet == "Healthy" ~ 1,
      Diet == "Moderate" ~ 2,
      Diet == "Unhealthy" ~ 3,
      TRUE ~ NA_real_
    )
  )


names(data_reduced)
# Set response and predictors
y <- data_reduced$Depression  # replace 'Depression' with your outcome variable if needed
X <- model.matrix(Depression ~ Acapressure + Sleep + Diet, data = data_reduced)[,-1]  # matrix of predictors (drop intercept)


set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)

X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]


#### lasso
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial")
plot(cv_lasso)

# Best lambda value (penalty strength)
best_lambda_lasso <- cv_lasso$lambda.min
print(best_lambda_lasso)

# Coefficients at best lambda
coef(cv_lasso, s = "lambda.min")


#### ridge
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0, family = "binomial")
plot(cv_ridge)

# Best lambda value
best_lambda_ridge <- cv_ridge$lambda.min
print(best_lambda_ridge)

# Coefficients at best lambda
coef(cv_ridge, s = "lambda.min")



# 1SE lambda values
lambda_1se_lasso <- cv_lasso$lambda.1se
lambda_1se_ridge <- cv_ridge$lambda.1se

lambda_1se_lasso
lambda_1se_ridge

coef(cv_lasso, s = "lambda.1se")
coef(cv_ridge, s = "lambda.1se")

# Non-zero coefficients (Lasso)
coef(cv_lasso, s = "lambda.min")[coef(cv_lasso, s = "lambda.min") != 0]



### test
# Lasso predictions (probabilities)
pred_lasso <- predict(cv_lasso, newx = X_test, s = "lambda.min", type = "response")
pred_lasso_class <- ifelse(pred_lasso > 0.5, 1, 0)  # Classify probabilities with 0.5 threshold
 
# Ridge predictions (probabilities)
pred_ridge <- predict(cv_ridge, newx = X_test, s = "lambda.min", type = "response")
pred_ridge_class <- ifelse(pred_ridge > 0.5, 1, 0)  # Classify probabilities with 0.5 threshold
 
# Accuracy for Lasso
accuracy_lasso <- mean(pred_lasso_class == y_test)
 
# Accuracy for Ridge
accuracy_ridge <- mean(pred_ridge_class == y_test)
 
# AUC for Lasso
auc_lasso <- roc(y_test, c(pred_lasso))$auc
 
# AUC for Ridge
auc_ridge <- roc(y_test, c(pred_ridge))$auc

# Print evaluation metrics
cat("Lasso Accuracy:", accuracy_lasso, "\n")
cat("Ridge Accuracy:", accuracy_ridge, "\n")
cat("Lasso AUC:", auc_lasso, "\n")
cat("Ridge AUC:", auc_ridge, "\n")


coef(cv_lasso, s = "lambda.min")
coef(cv_ridge, s = "lambda.min")