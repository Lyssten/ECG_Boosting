library(tidyverse)
library(caret)
library(gbm)
library(pROC)

# Load the dataset
data <- read.delim("Path/to/Common dataset.txt", sep="\t")

# Split dataset into features and target
X <- data %>% select(-class)
y <- data$class

# Apply StandardScaler
X_scaled <- as.data.frame(scale(X))

# Redefine target variable for binary classification
y_binary <- ifelse(y != 'N', 1, 0)

# Split the data for binary classification
set.seed(42)
splitIndex <- createDataPartition(y_binary, p=0.8, list=FALSE)
X_train_bin <- X_scaled[splitIndex, ]
y_train_bin <- y_binary[splitIndex]
X_test_bin <- X_scaled[-splitIndex, ]
y_test_bin <- y_binary[-splitIndex]

# Train the Stochastic Gradient Boosting model for binary classification
sgb_bin <- gbm.fit(X_train_bin, y_train_bin, distribution="bernoulli", n.trees=100, shrinkage=0.01, interaction.depth=4, bag.fraction=0.8)

# Evaluate the model on the testing set
y_pred_bin <- predict(sgb_bin, X_test_bin, n.trees=100, type="response")
y_pred_class <- ifelse(y_pred_bin > 0.5, 1, 0)
accuracy_bin <- mean(y_pred_class == y_test_bin)
cat(sprintf("Accuracy: %.2f%%\n", accuracy_bin * 100))

# Classification report
confMatrix <- table(y_test_bin, y_pred_class)
print(confMatrix)

# ROC curve
roc_obj <- roc(y_test_bin, y_pred_bin)
# auc_obj <- auc(roc_obj)

# Plotting
par(mfrow=c(1,2))
plot(roc_obj, main=sprintf("Receiver Operating Characteristic (ROC)\nAUC = %.2f", auc(roc_obj)))
abline(h=0, v=1, col="navy", lwd=2, lty=2)
