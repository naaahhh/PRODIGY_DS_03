# PRODIGY_DS_03
# Bank Marketing Prediction Using Decision Tree Classifier

## Project Overview

This project aims to predict whether a customer will subscribe to a term deposit based on demographic and behavioral data using a Decision Tree Classifier. The dataset used for this analysis is the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

The dataset contains 45,211 rows and 17 columns representing different customer attributes and their responses to a marketing campaign. The final goal is to classify whether the customer will subscribe to a term deposit, denoted by the `deposit` column.

## Dataset

The dataset consists of 17 columns:

1. **age**: Customer's age (numeric).
2. **job**: Type of job (categorical).
3. **marital**: Marital status (categorical).
4. **education**: Level of education (categorical).
5. **default**: Has credit in default? (categorical: 'yes', 'no').
6. **balance**: Average yearly balance (numeric).
7. **housing**: Has housing loan? (categorical: 'yes', 'no').
8. **loan**: Has personal loan? (categorical: 'yes', 'no').
9. **contact**: Type of communication (categorical).
10. **day**: Last contact day of the month (numeric).
11. **month**: Last contact month of the year (categorical).
12. **duration**: Last contact duration (numeric).
13. **campaign**: Number of contacts during this campaign (numeric).
14. **pdays**: Days since the customer was last contacted (numeric).
15. **previous**: Number of contacts performed before this campaign (numeric).
16. **poutcome**: Outcome of the previous campaign (categorical).
17. **deposit**: Has the client subscribed a term deposit? (target variable, categorical).

## Files in the Repository

- **PRODIGY_DS_03.ipynb**: The Jupyter Notebook containing the analysis, data preprocessing steps, and implementation of the decision tree model.
- **README.md**: A guide to understanding the project and running the code.
- **bank-full.csv**: The dataset used in the project (should be downloaded separately from the UCI repository).

## Steps to Run the Project

### 1. Set up the environment
Make sure you have the following Python libraries installed:
```bash
pip install pandas scikit-learn matplotlib
```

### 2. Load the Data
You can load the dataset by downloading it from the [UCI Bank Marketing Dataset repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

### 3. Data Preprocessing
The notebook preprocesses the data by encoding categorical variables using `LabelEncoder` and splitting the dataset into training and testing sets using an 80/20 ratio.

### 4. Model Training
The decision tree classifier is trained on the dataset using the `DecisionTreeClassifier` from `sklearn`. The notebook contains code to fit the model on the training data and evaluate it using accuracy and classification metrics.

### 5. Model Evaluation
The model's performance is evaluated using the following:
- **Accuracy**: Percentage of correctly predicted instances.
- **Classification Report**: Displays precision, recall, and F1 score for both classes (whether the customer subscribes or not).
- **Confusion Matrix**: Displays the performance of the classification model.

### 6. Decision Tree Visualization
You can visualize the decision tree structure using `matplotlib` and `sklearn.tree.plot_tree()`.

### 7. Additional Features
- Hyperparameter tuning using `GridSearchCV` to improve the decision tree’s performance.
- Handling class imbalance by adjusting weights or resampling the dataset.

## Example Output

- **Accuracy**: Around 87% accuracy using the default Decision Tree Classifier.
- **Classification Report**: Precision and recall for predicting both deposit classes.

## Future Improvements

- Implementing hyperparameter tuning for better generalization.
- Using more advanced models such as Random Forest, XGBoost, or Logistic Regression for comparison.
- Exploring feature importance and reducing dimensionality.
- Addressing the class imbalance in the dataset for better prediction of the minority class.

## License

This project is open-source and free to use. You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

