# ❤️ Heart Disease Prediction Using Decision Tree

## Project Overview
This project predicts whether a patient has heart disease using clinical data and a Decision Tree machine learning algorithm. It covers data preprocessing, exploratory data analysis, model training, evaluation, and visualization of the results.

## Features
- Data preprocessing and feature scaling
- Exploratory Data Analysis (EDA) with visualizations
- Decision Tree classification model
- Model evaluation with accuracy, confusion matrix, and classification report
- Visualization of the decision tree structure and feature importance

## Getting Started

### Prerequisites
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
 
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
## Dataset
Download the heart disease dataset <a href="https://github.com/LakshmiNarayana8464/Heart-Disease-Prediction-Using-Decision-Tree/blob/main/heart-disease.csv">heart.csv</a> and place it in the project directory. You can find datasets like the UCI Heart Disease Dataset.

## How to Run

Clone this repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```
Run the Jupyter Notebook or Python script:

For Jupyter Notebook:
```bash
jupyter notebook heart_disease_decision_tree.ipynb
```
Or run the Python script:

```bash
python heart_disease_decision_tree.py
```
Sample Code Snippet
```bash
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Assume X_train, y_train, X_test, y_test are already prepared

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
# Results
## Model Accuracy: 85%

Visualizations include:
- Heart disease class distribution
- Feature correlation heatmap
- Confusion matrix
- Decision tree plot
- Feature importance bar chart

# License
This project is licensed under the MIT License.

Feel free to customize the GitHub URL and dataset details.
