# Happy customers

## Project Overview

This project focuses on predicting customer satisfaction (specifically, dissatisfaction) for a logistics and delivery startup. The goal is to identify key factors influencing customer happiness and develop a predictive model to flag potentially unhappy customers. This allows for proactive intervention and improved operational efficiency.

## Dataset

The dataset used in this project comes from a survey. Preprocessing steps are included in the notebook to clean and prepare the data for modeling.

The dataset includes the following attributes:

* **Y:** Target variable, indicating customer satisfaction (0 = unhappy, 1 = happy).
* **X1:** My order was delivered on time (1-5 scale, 1 = less, 5 = more).
* **X2:** Contents of my order was as I expected (1-5 scale).
* **X3:** I ordered everything I wanted to order (1-5 scale).
* **X4:** I paid a good price for my order (1-5 scale).
* **X5:** I am satisfied with my courier (1-5 scale).
* **X6:** The app makes ordering easy for me (1-5 scale).

## Methodology

The following steps were taken:

1.  **Exploratory Data Analysis (EDA):** Understanding the data distribution and relationships between variables.
2.  **Model Selection:**
    * LazyPredictor was used to identify promising classification models.
    * BernoulliNB, AdaBoost, and LinearSVC were selected for further evaluation.
3.  **Feature Selection:**
    * Recursive Feature Elimination with Cross-Validation ([RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)) was used to identify the most influential predictors.
    * Leave-one-out cross validation was used due to the small size of the dataset.
4.  **Hyperparameter Optimization:**
    * [HyperOpt](http://hyperopt.github.io/hyperopt/) toolkit along with leave-one-out cross validation was used to optimize the hyperparameters of the selected models.
5.  **Ensemble Methods:**
    * Voting and stacking ensemble techniques were implemented to improve predictive performance.
6.  **Model Evaluation:**
    * Comprehensive evaluation metrics (accuracy, recall, precision, F1-score) were used to assess model performance.

## Repository Contents

* `data/`: Contains the dataset used in the project.
* `main.ipynb`: Jupyter notebook containing data analysis and model development.
* `models.py`: Functions for hyperparameter optimization using HyperOpt.
* `EDA/`: More results from EDA.
* `README.md`: This file.
* `requirements.txt`: List of Python dependencies.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/buddhiW/E02fXDJdn1pt6m5n
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Open and run `main.ipynb` to reproduce the analysis.

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