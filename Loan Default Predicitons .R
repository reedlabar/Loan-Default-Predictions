# Load necessary libraries
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)
library(car)
library(tidyverse)
library(randomForest)
library(class)
library(readr)
library(neuralnet)
library(gridExtra)
library(ROSE)
library(DMwR2)
library(smotefamily)


# Load the dataset
loan_data <- read_csv("/Users/reedlabar/Documents/Business Analytics /690 Project/690 Loan data Updated.csv")


#####Data Preprocessing####


colSums(is.na(loan_data))
summary(loan_data)

#identifying variables with missing values (only get thier names)

missing_vars <- colnames(loan_data)[colSums(is.na(loan_data)) > 0]
print(missing_vars)


# double check before you proceeed with your model
print(anyNA(loan_data))

#Formula
count_outliers <- function(column) {
  if(is.numeric(column)) { # check if the column is numeric
    Q1 <- quantile(column, 0.25, na.rm = TRUE)
    Q3 <- quantile(column, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    
    # count how many values fall outside of the IQR bounds
    sum(column < lower_bound | column > upper_bound, na.rm = TRUE)
  } else {
    NA # return NA for non numerical columns
  }
}

# Applying the function accros all columns to cound outliers 
outliers_counts <- sapply(loan_data,count_outliers)

#Displaying the number of outliers per column
print(outliers_counts)

# A vector of columns identified to have outliers
columns_with_outliers <- c(
  "loan_amnt", "funded_amnt", "funded_amnt_inv", "installment", "annual_inc", 
  "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", 
  "total_acc", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", 
  "total_rec_prncp", "total_rec_late_fee", "total_rec_int", "recoveries", 
  "collection_recovery_fee", "last_pymnt_amnt"
)

remove_outliers <- function(loan_data, column_name) {
  Q1 <- quantile(loan_data[[column_name]], 0.25, na.rm = TRUE)
  Q3 <- quantile(loan_data[[column_name]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  # count how many values fall outside of the IQR bounds
  subset(loan_data, data[[column_name]] >= lower_bound & data[[column_name]] <= upper_bound)
}

#interatively apply the function to each column indentified to have outliers
for(column in columns_with_outliers) {
  data <- remove_outliers(loan_data, column)
}

print(sprintf("Number of Rows after removing outliers: %d", nrow(data)))

print(sprintf("Number of Rows after before removing outliers: %d", nrow(loan_data)))


cap_outliers <- function(loan_data, column_name) {
  Q1 <- quantile(loan_data[[column_name]], 0.25, na.rm = TRUE)
  Q3 <- quantile(loan_data[[column_name]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  upper_bound <- Q3 + 1.5 * IQR
  lower_bound <- Q1 - 1.5 * IQR
  
  # Cap Values
  loan_data[[column_name]] <- ifelse(loan_data[[column_name]] > upper_bound, upper_bound,
                                     ifelse(loan_data[[column_name]] < lower_bound, lower_bound,
                                            loan_data[[column_name]]))
  return(loan_data)
}

#Applying the capping function to each column with outliers 
for(column in columns_with_outliers) {
  loan_data <- cap_outliers(loan_data, column)
}

# the data now has outliers capped at the upper and lower bounds

nrow(columns_with_outliers)

outlier_counts <- sapply(loan_data,count_outliers)

#displaying the number of outliers in each column
print(outlier_counts)





## Categorize 'delinq_2yrs' into specified ranges
loan_data$delinq_2yrs_category <- ifelse(loan_data$delinq_2yrs == 0, '0',
                                         ifelse(loan_data$delinq_2yrs <= 2, '1-2',
                                                ifelse(loan_data$delinq_2yrs <= 4, '3-4', '5+')))

# Select necessary columns for analysis
data_for_analysis <- loan_data %>%
  select(loan_amnt, delinq_2yrs, delinq_2yrs_category, int_rate, annual_inc, loan_status) %>%
  filter(!is.na(loan_amnt) & !is.na(delinq_2yrs) & !is.na(int_rate) & !is.na(annual_inc) & !is.na(loan_status))

# Convert 'delinq_2yrs_category' and 'loan_status' to factors
data_for_analysis$delinq_2yrs_category <- as.factor(data_for_analysis$delinq_2yrs_category)
data_for_analysis$loan_status <- as.factor(data_for_analysis$loan_status)

# Correlation Analysis
# Select necessary columns for analysis
numeric_columns <- data_for_analysis %>%
  select(loan_amnt, delinq_2yrs, int_rate, annual_inc)

# Convert 'int_rate' to numeric (if it's a percentage stored as a string)
numeric_columns$int_rate <- as.numeric(gsub("%", "", numeric_columns$int_rate)) / 100

# Calculate the correlation matrix
cor_matrix <- cor(numeric_columns, use = "complete.obs")

# Plot the correlation matrix
corrplot(cor_matrix, method = "circle", type = "lower", tl.col = "black", tl.srt = 45)

# Display correlation values
print(cor_matrix)

# Check the structure of the data
str(data_for_analysis)

# Check if there are any missing values in the relevant columns
missing_values <- colSums(is.na(data_for_analysis[c("loan_amnt", "delinq_2yrs", "int_rate", "annual_inc")]))
print(missing_values)

# Ensure no missing values in relevant columns
if (any(missing_values > 0)) {
  stop("There are missing values in the relevant columns.")
}

# Build a linear model using relevant predictors
vif_model <- lm(loan_amnt ~ delinq_2yrs + int_rate + annual_inc, data = data_for_analysis)

# Calculate VIF
vif_values <- vif(vif_model)

# Display VIF values
# Display VIF values
print(vif_values)

# Extract the GVIF values and calculate the VIF for each predictor
vif_values_extracted <- vif_values[, "GVIF"]

# Check if vif_values_extracted has the expected length and content
if (length(vif_values_extracted) > 0) {
  vif_data <- data.frame(
    Variable = names(vif_values_extracted),
    VIF = vif_values_extracted
  )
  
  # Display VIF data frame
  print(vif_data)
} else {
  stop("VIF values are not computed correctly or are empty.")
}

ggplot(vif_data, aes(x = reorder(Variable, VIF), y = VIF)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = 10, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(title = "Variance Inflation Factor (VIF) Values", x = "Predictors", y = "VIF") +
  theme_minimal()


# Create delinq_2yrs_category
loan_data$delinq_2yrs_category <- ifelse(loan_data$delinq_2yrs == 0, '0',
                                         ifelse(loan_data$delinq_2yrs <= 2, '1-2',
                                                ifelse(loan_data$delinq_2yrs <= 4, '3-4', '5+')))
loan_data$Loan <- loan_data$loan_status %in% c("Fully Paid", "Current")
loan_data$Deny <- loan_data$loan_status == "Charged Off"

# Check for the presence of delinq_2yrs_category and other necessary columns
if(!"delinq_2yrs_category" %in% colnames(loan_data)) {
  stop("The column 'delinq_2yrs_category' does not exist in the dataset.")
}
# Convert necessary columns to numeric, handling non-numeric values gracefully
loan_data$loan_amnt <- as.numeric(gsub("[^0-9.]", "", loan_data$loan_amnt))
loan_data$int_rate <- as.numeric(gsub("[^0-9.]", "", loan_data$int_rate))
loan_data$annual_inc <- as.numeric(gsub("[^0-9.]", "", loan_data$annual_inc))

# Check for NA values in the critical columns
na_summary <- colSums(is.na(loan_data %>% select(loan_amnt, int_rate, annual_inc, delinq_2yrs_category)))
print(na_summary)

# Remove rows with NA values in critical columns
loan_data <- loan_data %>%
  filter(!is.na(loan_amnt) & !is.na(int_rate) & !is.na(annual_inc) & !is.na(delinq_2yrs_category))

# Ensure the target variable has enough observations in each category
table(loan_data$delinq_2yrs_category)

# Undersampling
data_undersampled <- ovun.sample(delinq_2yrs_category ~ ., data = loan_data,
                                 method = "under", p = 0.5, seed = 1)$data
table(data_undersampled$delinq_2yrs_category)



#######Distribution of Classes##########


# Ensure 'loan_status' and 'delinq_2yrs_category' are factors
loan_data$loan_status <- as.factor(loan_data$loan_status)
loan_data$delinq_2yrs_category <- ifelse(loan_data$delinq_2yrs == 0, '0',
                                         ifelse(loan_data$delinq_2yrs <= 2, '1-2',
                                                ifelse(loan_data$delinq_2yrs <= 4, '3-4', '5+')))
loan_data$delinq_2yrs_category <- as.factor(loan_data$delinq_2yrs_category)

# Ensure the numerical columns are treated as numeric
loan_data$loan_amnt <- as.numeric(loan_data$loan_amnt)
loan_data$int_rate <- as.numeric(gsub("%", "", loan_data$int_rate)) / 100
loan_data$annual_inc <- as.numeric(loan_data$annual_inc)

# Plot distribution of loan amounts by loan status
p1 <- ggplot(loan_data, aes(x = loan_amnt, fill = loan_status)) +
  geom_histogram(binwidth = 1000, position = "dodge", alpha = 0.7) +
  labs(title = "Distribution of Loan Amounts by Loan Status", x = "Loan Amount", y = "Count") +
  theme_minimal()

# Plot distribution of interest rates by loan status
p2 <- ggplot(loan_data, aes(x = int_rate, fill = loan_status)) +
  geom_histogram(binwidth = 0.01, position = "dodge", alpha = 0.7) +
  labs(title = "Distribution of Interest Rates by Loan Status", x = "Interest Rate", y = "Count") +
  theme_minimal()

# Plot distribution of annual incomes by loan status
p3 <- ggplot(loan_data, aes(x = annual_inc, fill = loan_status)) +
  geom_histogram(binwidth = 10000, position = "dodge", alpha = 0.7) +
  labs(title = "Distribution of Annual Incomes by Loan Status", x = "Annual Income", y = "Count") +
  theme_minimal()

# Plot distribution of delinquencies by loan status
p4 <- ggplot(loan_data, aes(x = delinq_2yrs_category, fill = loan_status)) +
  geom_bar(position = "dodge", alpha = 0.7) +
  labs(title = "Distribution of Delinquencies by Loan Status", x = "Delinquencies in 2 Years", y = "Count") +
  theme_minimal()

# Arrange plots in a grid
grid.arrange(p1, p2, p3, p4, nrow = 2)





#####Linear Model######
linear_model <- lm(loan_amnt ~ delinq_2yrs_category + int_rate + annual_inc, data = data_for_analysis)
summary(linear_model)

# Predict values
predicted_values <- predict(linear_model)

# Predicted vs Actual Loan Amounts with Outliers
ggplot(data_for_analysis, aes(x = loan_amnt, y = predicted_values)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted Loan Amounts", x = "Actual Loan Amount", y = "Predicted Loan Amount")


# Calculate residuals
residuals <- resid(linear_model)

# Identify outliers based on residuals
threshold <- 3 * sd(residuals)
outliers <- abs(residuals) > threshold

# Remove outliers
filtered_data <- data_for_analysis[!outliers, ]

# Refit the linear model without outliers
linear_model_filtered <- lm(loan_amnt ~ delinq_2yrs_category + int_rate + annual_inc, data = filtered_data)
summary(linear_model_filtered)

# Predict values without outliers
predicted_values_filtered <- predict(linear_model_filtered)

# Plot actual vs predicted values without outliers
ggplot(filtered_data, aes(x = loan_amnt, y = predicted_values_filtered)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted Loan Amounts (Outliers Removed)", x = "Actual Loan Amount", y = "Predicted Loan Amount")

# Plot residuals without outliers
residuals_filtered <- resid(linear_model_filtered)

ggplot(filtered_data, aes(x = predicted_values_filtered, y = residuals_filtered)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Predicted Values (Outliers Removed)", x = "Predicted Loan Amount", y = "Residuals")






#KNN
#####k-NN#####


# Data Preprocessing
loan_data$delinq_2yrs_category <- ifelse(loan_data$delinq_2yrs == 0, '0',
                                         ifelse(loan_data$delinq_2yrs <= 2, '1-2',
                                                ifelse(loan_data$delinq_2yrs <= 4, '3-4', '5+')))

# Ensure the necessary columns are numeric
loan_data$int_rate <- as.numeric(gsub("%", "", loan_data$int_rate)) / 100
loan_data$annual_inc <- as.numeric(loan_data$annual_inc)

# Select necessary columns for analysis
data_for_analysis <- loan_data %>%
  select(loan_amnt, delinq_2yrs, delinq_2yrs_category, int_rate, annual_inc, loan_status) %>%
  filter(!is.na(loan_amnt) & !is.na(delinq_2yrs_category) & !is.na(loan_status))

# Convert 'delinq_2yrs_category' and 'loan_status' to factors
data_for_analysis$delinq_2yrs_category <- as.factor(data_for_analysis$delinq_2yrs_category)
data_for_analysis$loan_status <- as.factor(data_for_analysis$loan_status)

# Ensure all columns selected for normalization are numeric
numeric_features <- data_for_analysis %>%
  select(loan_amnt, delinq_2yrs, int_rate, annual_inc) %>%
  mutate(across(everything(), as.numeric))  # Ensure all columns are numeric

# Check the structure to confirm all are numeric
str(numeric_features)

# Normalize the numeric features
normalized_features <- as.data.frame(scale(numeric_features))

# Encode categorical features
encoded_features <- model.matrix(~ delinq_2yrs_category - 1, data = data_for_analysis)

# Combine normalized numeric features and encoded categorical features
knn_data <- cbind(normalized_features, encoded_features)

# Add the target variable to the dataset
knn_data$loan_status <- data_for_analysis$loan_status

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(knn_data$loan_status, p = 0.7, list = FALSE)
train_data <- knn_data[trainIndex, ]
test_data <- knn_data[-trainIndex, ]

# Define predictors and target for training and testing sets
train_predictors <- train_data %>%
  select(-loan_status)
test_predictors <- test_data %>%
  select(-loan_status)
train_target <- train_data$loan_status
test_target <- test_data$loan_status

# Fit k-NN model and make predictions
k <- 5  # Choose the number of neighbors
knn_predictions <- knn(train = train_predictors, test = test_predictors, cl = train_target, k = k)

# Confusion matrix
confusion_matrix <- confusionMatrix(knn_predictions, test_target)
print(confusion_matrix)

# Plot actual vs predicted values
plot_data <- data.frame(
  Actual = test_target,
  Predicted = knn_predictions
)

ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_jitter(alpha = 0.5) +
  labs(title = "Actual vs Predicted Loan Status (k-NN)", x = "Actual Loan Status", y = "Predicted Loan Status") +
  theme_minimal()








# Random Forest Model
#####Random Forest######
set.seed(123)
random_forest_model <- randomForest(loan_status ~ loan_amnt + delinq_2yrs_category + int_rate + annual_inc, data = data_for_analysis, ntree = 100, importance = TRUE)
print(random_forest_model)

# Plot the importance of variables
varImpPlot(random_forest_model, main = "Variable Importance in Random Forest")

# Predictions
rf_predictions <- predict(random_forest_model, newdata = data_for_analysis)
confusionMatrix(as.factor(rf_predictions), data_for_analysis$loan_status)

# Plot actual vs predicted values
plot_data <- data.frame(
  Actual = data_for_analysis$loan_status,
  Predicted = rf_predictions
)

ggplot(plot_data, aes(x = as.factor(Actual), y = as.factor(Predicted))) +
  geom_jitter(alpha = 0.5) +
  labs(title = "Actual vs Predicted Loan Status", x = "Actual Loan Status", y = "Predicted Loan Status") +
  theme_minimal()


#Neural Net 
####Neural Net######
#Neural Net 

# Create binary variables for loan approval and denial
loan_data$Loan <- loan_data$loan_status %in% c("Fully Paid", "Current")
loan_data$Deny <- loan_data$loan_status == "Charged Off"

# Check for the presence of delinq_2yrs_category and other necessary columns
if(!"delinq_2yrs_category" %in% colnames(loan_data)) {
  stop("The column 'delinq_2yrs_category' does not exist in the dataset.")
}

# Create delinq_2yrs_category
loan_data$delinq_2yrs_category <- ifelse(loan_data$delinq_2yrs == 0, '0',
                                         ifelse(loan_data$delinq_2yrs <= 2, '1-2',
                                                ifelse(loan_data$delinq_2yrs <= 4, '3-4', '5+')))

# Convert necessary columns to numeric, handling non-numeric values gracefully
loan_data$loan_amnt <- as.numeric(gsub("[^0-9.]", "", loan_data$loan_amnt))
loan_data$int_rate <- as.numeric(gsub("[^0-9.]", "", loan_data$int_rate))
loan_data$annual_inc <- as.numeric(gsub("[^0-9.]", "", loan_data$annual_inc))

# Check for NA values in the critical columns
na_summary <- colSums(is.na(loan_data %>% select(loan_amnt, int_rate, annual_inc, delinq_2yrs_category)))
print(na_summary)

# Handle NA values (either remove or impute)
# Here we will impute missing values with the median of each column for demonstration purposes
loan_data$loan_amnt[is.na(loan_data$loan_amnt)] <- median(loan_data$loan_amnt, na.rm = TRUE)
loan_data$int_rate[is.na(loan_data$int_rate)] <- median(loan_data$int_rate, na.rm = TRUE)
loan_data$annual_inc[is.na(loan_data$annual_inc)] <- median(loan_data$annual_inc, na.rm = TRUE)
loan_data$delinq_2yrs_category[is.na(loan_data$delinq_2yrs_category)] <- median(as.numeric(as.factor(loan_data$delinq_2yrs_category)), na.rm = TRUE)

# Re-create binary variables for loan approval and denial to ensure no NAs
loan_data$Loan <- as.numeric(loan_data$loan_status %in% c("Fully Paid", "Current"))
loan_data$Deny <- as.numeric(loan_data$loan_status == "Charged Off")

# Select relevant columns
ann_loan <- loan_data %>% select(loan_amnt, int_rate, annual_inc, delinq_2yrs_category, Loan, Deny)

# Convert delinq_2yrs_category to numeric (ensure it's a factor first, then numeric)
ann_loan$delinq_2yrs_category <- as.numeric(as.factor(ann_loan$delinq_2yrs_category))

# Verify the final dataset structure
str(ann_loan)

# Set a random seed for reproducibility in neural network training
set.seed(1)

# Train a neural network with 'Loan' and 'Deny' as output
# and 'annual_inc', 'delinq_2yrs_category', 'loan_amnt', and 'int_rate' as input features
# Specify three hidden neurons, and use a logistic activation function (linear.output = FALSE)
nn <- neuralnet(Loan + Deny ~ annual_inc + delinq_2yrs_category + loan_amnt + int_rate, data = ann_loan, linear.output = FALSE, hidden = 3)

# Train the neural network again with a threshold for stopping the training when the change in error is less than 0.1
nn <- neuralnet(Loan + Deny ~ annual_inc + delinq_2yrs_category + loan_amnt + int_rate, data = ann_loan, linear.output = FALSE, hidden = 3, threshold = 0.1)


# Use the trained neural network model 'nn' to compute outputs for the given inputs 'annual_inc', 'delinq_2yrs_category', 'loan_amnt', 'int_rate'
predict <- compute(nn, ann_loan %>% select(annual_inc, delinq_2yrs_category, loan_amnt, int_rate))

predicted_probs <- predict$net.result

# Convert the probabilities to binary outcomes based on a threshold (e.g., 0.5)
predicted_classes <- ifelse(predicted_probs[,1] > 0.5, 1, 0) # Ensure binary outcomes are 1 and 0

# Actual Classes 
actual_classes <- ann_loan$Loan

# Ensure levels match between predicted and actual classes
predicted_classes <- factor(predicted_classes, levels = c(0, 1))
actual_classes <- factor(actual_classes, levels = c(0, 1))

# Generate confusion matrix
conf_matrix <- confusionMatrix(predicted_classes, actual_classes)
print(conf_matrix)

# Print neural network model summary
print(nn)
print(nn$call)

