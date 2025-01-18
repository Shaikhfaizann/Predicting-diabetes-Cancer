# Predicting-diabetes-Cancer

This repository contains a Python-based project focused on predicting breast cancer diagnoses using machine learning techniques. The dataset utilized in this project is the Breast Cancer Wisconsin dataset, which is loaded using `sklearn`. The project involves exploratory data analysis (EDA), data visualization, and the implementation of classification algorithms to build predictive models.

Features:
1. Exploratory Data Analysis (EDA):
   - Detailed data inspection with `pandas` and `seaborn`.
   - Visualization of feature distributions and pairwise relationships.

2. Data Preprocessing:
   - Splitting data into training and testing sets.
   - Scaling features using various scalers (`StandardScaler`, `MinMaxScaler`, etc.) to evaluate their impact.

3. Model Implementation:
   - Logistic Regression: Baseline classification model.
   - K-Nearest Neighbors (KNN): Fine-tuned using hyperparameter optimization (number of neighbors).

4. Performance Metrics:
   - Evaluation using metrics such as confusion matrix, classification report, and recall scores to address imbalanced data challenges.
   - Graphical error rate analysis for hyperparameter tuning.

5. Key Insights:
   - Comparison of model performance across different preprocessing techniques.
   - Insights into overfitting and strategies for mitigation.

 Tools and Libraries:
Data Manipulation and Visualization: `pandas`, `numpy`, `matplotlib`, `seaborn`
Machine Learning `scikit-learn`

Usage:
Clone the repository and run the Jupyter Notebook or `.py` scripts to explore the analysis and reproduce results. You can modify hyperparameters and preprocessing steps to experiment with the model's performance.


