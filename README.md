# Happy customers

## Project Overview

This repository contains a classification model built for a survey dataset. The project involves selecting and evaluating different machine learning classifiers to determine the best-performing model.

## Model Selection

Based on LazyPredictor results, three classifiers were chosen:

- **BernoulliNB**: A probabilistic model based on Naive Bayes.
- **AdaBoost**: An ensemble method that combines weak classifiers to improve accuracy.
- **LinearSVC**: A linear model using Support Vector Classification.

## Feature Selection

Feature selection is a crucial step in this project to improve model performance and reduce overfitting. Various techniques such as correlation analysis, mutual information, and recursive feature elimination (RFE) were explored to identify the most relevant features. The selected features significantly impact the classifier’s ability to generalize well on new data.

## Installation

To run this project, install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter Notebook to execute the classification pipeline:

```bash
jupyter notebook main.ipynb
```

## Dataset

The dataset used in this project comes from a survey. Preprocessing steps are included in the notebook to clean and prepare the data for modeling.

## Results

The performance of the selected classifiers is evaluated based on multiple metrics, including accuracy, precision, recall, and F1-score. A comparison table is included in the notebook to illustrate the strengths and weaknesses of each model.

- **BernoulliNB**: Performed well on categorical data but struggled with complex relationships between features.
- **AdaBoost**: Showed strong generalization capabilities and performed well in cases with imbalanced data.
- **LinearSVC**: Provided a balance between efficiency and accuracy but was sensitive to feature scaling.

Additionally, confusion matrices and classification reports are included to analyze misclassifications and potential areas for improvement. The final model was selected based on a trade-off between interpretability, computational efficiency, and classification performance.


