# Happy customers

## Project Overview

This repository contains a classification model built for a survey dataset. The project involves selecting and evaluating different machine learning classifiers to determine the best-performing model.

## Model Selection

Based on LazyPredictor results, three classifiers were chosen:

- **BernoulliNB**: A probabilistic model based on Naive Bayes.
- **AdaBoost**: An ensemble method that combines weak classifiers to improve accuracy.
- **LinearSVC**: A linear model using Support Vector Classification.

## Feature Selection

Feature selection is a crucial step in this project to improve model performance and reduce overfitting. Various techniques exhaustive feature search and recursive feature elimination (RFE) were explored to identify the most relevant features. The selected features significantly impact the classifier’s ability to generalize well on new data.

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

## Conclusion:

My analysis aimed to predict unhappy customers (unhappy = positive class), since focusing on dissatisfied is more crucial at this stage. **It's important to acknowledge that this analysis was conducted on a relatively small dataset, which inherently limits the generalizability and stability of the findings.** I explored various classification models, feature selection techniques, and ensemble methods to optimize predictive performance. Out of 30 classification models, 3 were selected based on baseline recall metrics. While the Naive Bayes and LinearSVC models demonstrated limited effectiveness, the AdaBoost classifier emerged as the most robust individual model, particularly after rigorous feature selection using Recursive Feature Elimination with Leave-One-Out Cross-Validation.

My findings underscore the significance of specific factors—on-time delivery (X1), order completeness (X3), courier satisfaction (X5), and app usability (X6)—in reducing customer dissatisfaction. Notably, ensemble methods, specifically voting and stacking, significantly improved predictive accuracy and recall compared to individual models. This highlights the value of combining diverse model architectures to capture complex data patterns and enhance overall prediction capabilities. **However, the small dataset size means that these patterns may not be fully representative of the broader customer base.**

## Recommendations:

1.  **Prioritize Operational Improvements in Key Areas:**
    * The strong correlation between dissatisfaction and factors X1, X3, X5, and X6 necessitates focused operational improvements.
    * **X1 (On-Time Delivery):** Implement real-time tracking, optimize delivery routes, and provide proactive communication regarding delays. Invest in technology to improve delivery time predictability.
    * **X3 (Order Completeness):** Enhance inventory management, improve order verification processes, and ensure clear communication regarding item availability.
    * **X5 (Courier Satisfaction):** Invest in courier training, provide competitive compensation, and foster a supportive work environment. Happy couriers lead to happy customers.
    * **X6 (App Usability):** Conduct regular user testing, simplify the ordering process, and address any technical glitches promptly.
2.  **Leverage the AdaBoost Ensemble Model for Prediction:**
    * The AdaBoost ensemble model, enhanced by voting or stacking, demonstrated the highest predictive performance. Integrate this model into the customer feedback analysis pipeline.
    * Use the model to create a early warning system to flag customers that are likely to be unhappy. 
3.  **Prioritize Dataset Expansion:**
    * Implement strategies to gather feedback from a wider range of customers, including those who may not typically respond to surveys.
    * Explore methods such as in-app feedback prompts, post-delivery surveys, and social media monitoring to increase data collection.
4.  **Invest in Feature Engineering:**
    * While the current features provide valuable insights, explore additional data sources and feature engineering techniques to further enhance predictive accuracy.
    * Consider factors such as delivery distance, order frequency, and customer demographics.
6.  **Focus on "Unhappy" Customer Recovery:**
    * Develop targeted strategies for addressing and resolving the issues identified by the predictive model.
    * Proactive outreach to potentially unhappy customers can mitigate negative experiences and build customer loyalty.