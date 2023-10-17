library(tidyverse)
library(caret)
library(gbm)
library(pROC)

# Load the dataset
data <- read.delim("D:/programming/projects/py/ECG_elective/Dataset/Common dataset.txt", sep="\t")

# Split dataset into features and target
X <- data %>% select(-class)
y <- data$class

# Apply StandardScaler
X_scaled <- as.data.frame(scale(X))

# List of unique disease classes excluding 'N'
disease_classes <- unique(y[y != 'N'])

# Dictionary (list in R) to store metrics for each class pairing
metrics_list <- list()

# Iterate over each disease class to create a binary dataset and train a model
for (disease in disease_classes) {
  # Create binary dataset
  mask <- y %in% c('N', disease)
  X_binary <- X_scaled[mask, ]
  y_binary <- as.numeric(y[mask] != 'N')
  
  # Split the data
  set.seed(42)
  splitIndex <- createDataPartition(y_binary, p=0.8, list=FALSE)
  X_train_bin <- X_binary[splitIndex, ]
  y_train_bin <- y_binary[splitIndex]
  X_test_bin <- X_binary[-splitIndex, ]
  y_test_bin <- y_binary[-splitIndex]
  
  # Train the Stochastic Gradient Boosting model for binary classification
  sgb_bin <- gbm.fit(X_train_bin, y_train_bin, distribution="bernoulli", n.trees=100, shrinkage=0.01, interaction.depth=4, bag.fraction=0.8)
  
  # Predictions
  y_pred_bin <- predict(sgb_bin, X_test_bin, n.trees=100, type="response")
  
  # ROC curve
  roc_obj <- roc(y_test_bin, y_pred_bin)
  
  # Store metrics
  metrics_list[[disease]] <- list(
    accuracy = mean((y_pred_bin > 0.5) == y_test_bin),
    roc_obj = roc_obj
  )
}

print(metrics_list)

# Plotting ROC curves for each class pairing
par(mfrow=c(1,1))
plot(roc_obj, col=1, lwd=2, legacy.axes=TRUE)
colors <- rainbow(length(disease_classes))
for (i in seq_along(disease_classes)) {
  lines(metrics_list[[disease_classes[i]]]$roc_obj, col=colors[i], lwd=2)
}
abline(h=0, v=1, col="navy", lwd=2, lty=2)
legend("bottomright", legend=disease_classes, fill=colors)
