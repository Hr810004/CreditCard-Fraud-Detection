# Credit Card Fraud Detection using Custom Logistic Regression

## Project Overview

This project addresses the real-world problem of detecting fraudulent credit card transactions using a custom-built machine learning model implemented entirely from scratch, without relying on existing ML libraries. The goal is to classify transactions as legitimate or fraudulent based on transaction data.

The dataset used is a publicly available credit card transactions dataset with highly imbalanced classes (frauds are very rare). A logistic regression model with weighted loss for imbalanced classification is developed from the ground up, including preprocessing, training, threshold tuning, evaluation, and visualization.

## Features

- Custom implementation of Logistic Regression with:
  - Sigmoid activation
  - Weighted binary cross-entropy loss to address class imbalance
  - L2 regularization
  - Gradient descent optimization
- Data preprocessing including:
  - Handling missing values (if any)
  - Standardization of all features
  - Train-validation split
- Threshold tuning to maximize F1-score for the positive (fraud) class
- Comprehensive evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC AUC
- Visualizations:
  - Class distribution
  - Training and validation loss curves
  - Confusion matrix
  - ROC curve

## Project Structure

- `creditcard.ipynb`: Jupyter notebook containing the full implementation, explanations, and visualizations.
- Dataset: Credit card transactions dataset (assumed pre-downloaded or accessible).
- README.md: Project description and usage.

## Usage

1. Load dataset
2. Preprocess data (missing value imputation if needed, feature standardization)
3. Split data into train and validation sets
4. Train the custom logistic regression model with class weight applied for imbalance
5. Tune classification threshold to optimize F1-score
6. Evaluate final model performance on the validation set
7. View plots for class distribution, loss curves, confusion matrix, and ROC curve

## Methodology

- The logistic regression model is implemented from scratch using NumPy.
- To handle class imbalance, positive samples (fraud cases) are given a higher weight inversely proportional to their frequency.
- Binary cross-entropy loss is weighted accordingly with L2 regularization for model generalization.
- Model optimization is performed via gradient descent for a fixed number of epochs.
- Post training, optimal threshold tuning is done by scanning thresholds to maximize F1-score on validation set predictions.
- Evaluation includes standard classification metrics and ROC AUC for overall performance assessment.

## Results

- Positive class weight calculated based on train set imbalance ratio.
- Training and validation loss consistently decrease, showing good convergence.
- Optimal classification threshold found near 0.95 maximizing F1-score.
- Final validation set performance metrics:
  - Accuracy: 0.9994
  - Precision: 0.8431
  - Recall: 0.8190
  - F1-score: 0.8309
  - ROC AUC: 0.9804

## Insights and Challenges

- The dataset is highly imbalanced requiring special handling to avoid bias toward the majority class.
- Weighted loss combined with threshold tuning significantly improved detection of minority class.
- Regularization helped mitigate overfitting in this relatively simple linear model.
- The model demonstrates strong predictive ability for identifying frauds while maintaining low false positive rates.

## Future Work

- Experiment with other custom-built models such as decision trees or neural networks.
- Explore feature engineering and dimensionality reduction techniques.
- Implement ensemble methods like bagging or random forests from scratch for improved performance.

---

Feel free to reach out for questions or further assistance regarding this project implementation.
