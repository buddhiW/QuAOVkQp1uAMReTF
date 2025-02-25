#Imports
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from itertools import combinations
from datetime import datetime
from collections import defaultdict, Counter

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import RFE, RFECV, SelectKBest, mutual_info_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from sklearn.compose import ColumnTransformer

from hyperopt import fmin, tpe,rand, hp, STATUS_OK,space_eval, Trials
from hyperopt.pyll.base import scope
from hyperopt.early_stop import no_progress_loss

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC, NuSVC

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# Define early stopping function
def early_stop_fn(trials):
    threshold=5
    if len(trials.trials) < threshold:
        return False  # Not enough trials to compare

    # Extract loss values from past trials
    losses = [t['result']['loss'] for t in trials.trials if 'loss' in t['result']]
    
    if len(losses) < threshold:
        return False  # Ensure we have enough loss values

    best_loss_before = min(losses[:-threshold])  # Best loss before recent threshold
    best_loss_now = min(losses)  # Current best loss

    return best_loss_now >= best_loss_before  # Stop if no improvement

def display_results(model, model_name, x_train, y_train, x_test, y_test):
    
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=0)
    recall = recall_score(y_test, y_pred, pos_label=0)
    metrics = []
    metrics.append([accuracy, recall, precision])

    plt.figure(figsize=(3, 3))
    cm = confusion_matrix(y_test, y_pred)
    s = sns.heatmap(cm, cbar = False, annot=True, cmap='Blues')
    s.set_xlabel('Pred')
    s.set_ylabel('True')
    s.set_title(model_name)

    metrics_df = pd.DataFrame(metrics, columns=['Accuracy', 'Recall', 'Precision'])
    print(metrics_df)
    plt.show()



# Some Sklearn models do not have feature importance function.
# So, wrappers for those classifiers were created.
class MyBaggingClassifier(BaggingClassifier):
    @property
    def feature_importances_(self):
        feature_importances = np.mean([
            tree.feature_importances_ for tree in self.estimators_], axis=0)
        
        return feature_importances
    
class MyBernoulliNB(BernoulliNB):
    @property
    def feature_importances_(self):
        # Compute feature importance as absolute difference in log probabilities
        return np.abs(self.feature_log_prob_[1] - self.feature_log_prob_[0])

# Naive Bayes hyperparameter tuning with HyperOpt  
def hyperOpt_NaiveBayes(X_subset, Y, preprocessor, scorer, seed, cur_best_params=None):

    loo = LeaveOneOut()
     
    x_train_subset, x_test_subset, y_train, y_test = train_test_split(X_subset, Y, test_size=0.2, random_state=seed)

    # Define the objective function
    def objective(params):
        #print(f"Trying params: {params}")
        clf = MyBernoulliNB(**params)
        model = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", clf)
        ])

        score = cross_val_score(model, x_train_subset, y_train, cv=loo, scoring=scorer).mean()
        return {'loss': -score, 'status': STATUS_OK}

    # Define the search space
    space = {
        'alpha': hp.loguniform('alpha', -5, 2),  # alpha ∈ (exp(-5), exp(2)) ≈ (0.0067, 7.39)
        'fit_prior': hp.choice('fit_prior', [True, False]),  # Boolean choice
        'binarize': hp.uniform('binarize', 0, 1)  # Feature binarization threshold
    }
    # Run the optimization
    starttime = timer()
    best_params = fmin(objective, space, algo=rand.suggest, max_evals=500, early_stop_fn=no_progress_loss(10), rstate=np.random.default_rng(seed))
    timer(starttime)

    # Print the best hyperparameters
    best_params = space_eval(space, best_params)
    print("Best hyperparameters:", best_params)

    # Evaluate the model with the best hyperparameters
    final_clf = MyBernoulliNB(**best_params)
    final_model = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", final_clf)
    ])

    display_results(final_model, 'NaiveBayes', x_train_subset, y_train, x_test_subset, y_test)

    return best_params


# AdaBoost hyperparameter tuning with HyperOpt
def hyperOpt_AdaBoost(X_subset, Y, preprocessor, scorer, seed):

    loo = LeaveOneOut()
    x_train_subset, x_test_subset, y_train, y_test = train_test_split(X_subset, Y, test_size=0.2, random_state=seed)

    # Define the objective function
    def objective(params):
        #print(f"Trying params: {params}")

        # # Create the base estimator separately
        # estimator = DecisionTreeClassifier(
        #     max_depth=params.pop('max_depth'),
        #     min_samples_split=params.pop('min_samples_split'),
        #     random_state=seed
        # )

        clf = AdaBoostClassifier(**params, random_state=seed) #estimator=estimator,
    
        model = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", clf)
        ])

        score = cross_val_score(model, x_train_subset, y_train, cv=loo, scoring=scorer).mean()
        return {'loss': -score, 'status': STATUS_OK}

    # Define the search space
    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 10)),  # Number of boosting rounds
        'learning_rate': hp.loguniform('learning_rate', -4, 0),  # Learning rate (0.0001 to 1)
        'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R']),  # Type of boosting
        #'max_depth': scope.int(hp.quniform('base_max_depth', 1, 30, 1)),  # Depth of weak learners
        #'min_samples_split': hp.uniform('base_min_samples_split', 0.01, 0.5),  # Min samples per split
    }

    # Run the optimization
    best_params = fmin(objective, space, algo=tpe.suggest, max_evals=500, early_stop_fn=no_progress_loss(10), rstate=np.random.default_rng(seed))

    # Print the best hyperparameters
    best_params = space_eval(space, best_params)
    print("Best hyperparameters:", best_params)

    # Evaluate the model with the best hyperparameters
    best_params['n_estimators'] = int(best_params['n_estimators'])
    #best_params['max_depth'] = int(best_params['max_depth'])

    # estimator = DecisionTreeClassifier(
    #         max_depth=best_params.pop('max_depth'),
    #         min_samples_split=best_params.pop('min_samples_split'),
    #         random_state=seed
    #     )

    final_clf = AdaBoostClassifier(**best_params, random_state=seed) #estimator=estimator, 

    final_model = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", final_clf)
    ])

    display_results(final_model, 'AdaBoost', x_train_subset, y_train, x_test_subset, y_test)

    return best_params

    
# LinearSVC hyperparameter tuning with HyperOpt
def hyperOpt_LinearSVC(X_subset, Y, preprocessor, scorer,  seed):

    loo = LeaveOneOut()

    x_train_subset, x_test_subset, y_train, y_test = train_test_split(X_subset, Y, test_size=0.2, random_state=seed)

    # Define the objective function
    def objective(params):
        #print(f"Trying params: {params}")
        clf = LinearSVC(**params, random_state=seed)
        #clf = SVC(**params, kernel='linear', random_state=seed, probability=True)
        model = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", clf)
        ])

        score = cross_val_score(model, x_train_subset, y_train, cv=loo, scoring=scorer).mean()
        return {'loss': -score, 'status': STATUS_OK}

    # Define the search space
    space = {
        'C': hp.loguniform('C', -5, 2),  # Regularization strength (10^-5 to 10^2)
        'loss': hp.choice('loss', ['hinge', 'squared_hinge']),  # Loss function
        'tol': hp.loguniform('tol', -6, -2),  # Stopping tolerance (10^-6 to 10^-2)
        'max_iter': scope.int(hp.quniform('max_iter', 500, 5000, 100)),  # Number of iterations
    }

    # Run the optimization
    best_params = fmin(objective, space, algo=tpe.suggest, max_evals=500, early_stop_fn=no_progress_loss(10), rstate=np.random.default_rng(seed)) #rand

    # Print the best hyperparameters
    best_params = space_eval(space, best_params)
    print("Best hyperparameters:", best_params)

    # Evaluate the model with the best hyperparameters

    best_params['max_iter'] = int(best_params['max_iter'])

    final_clf = LinearSVC(**best_params, random_state=seed)
    #final_clf = SVC(**best_params, kernel='linear', random_state=seed, probability=True)
    final_model = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", final_clf)
    ])

    display_results(final_model, 'LinearSVC', x_train_subset, y_train, x_test_subset, y_test)

    return best_params
    

# LGBM optimization with HyperOpt
def hyperOpt_LGBM(X_subset, Y, preprocessor, scorer,  seed):
    
    loo = LeaveOneOut()
    x_train_subset, x_test_subset, y_train, y_test = train_test_split(X_subset, Y, test_size=0.2, random_state=seed)

    # Define the objective function
    def objective(params):
        #print(f"Trying params: {params}")
        model = LGBMClassifier(**params, verbose=-1)
        score = cross_val_score(model, x_train_subset, y_train, cv=loo, scoring=scorer).mean()
        return {'loss': -score, 'status': STATUS_OK}

    # Define the search space
    space = {
        'num_leaves': scope.int(hp.quniform('num_leaves', 2, 32, 2)),  # Small range to avoid overfitting
        'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 5, 50, 5)),  # Prevent too small leaves
        'learning_rate': hp.loguniform('learning_rate', -3, 0),  # 0.001 to 1.0
        'max_depth': hp.choice('max_depth', [-1, 3, 5, 7]),  # Limit depth for small dataset
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),  # Feature selection
        'subsample': hp.uniform('subsample', 0.5, 1.0),  # Row sampling for regularization
        'reg_alpha': hp.loguniform('reg_alpha', -4, 1),  # L1 regularization (0.0001 to 10)
        'reg_lambda': hp.loguniform('reg_lambda', -4, 1),  # L2 regularization (0.0001 to 10)
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),  # Limit estimators for small dataset
        'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'rf']),  # Try both methods
    }

    # Run the optimization
    best_params = fmin(objective, space, algo=tpe.suggest, max_evals=500, early_stop_fn=no_progress_loss(10), rstate=np.random.default_rng(seed))

    # Print the best hyperparameters
    best_params = space_eval(space, best_params)
    print("Best hyperparameters:", best_params)

    # Evaluate the model with the best hyperparameters
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['min_data_in_leaf'] = int(best_params['min_data_in_leaf'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    final_clf = LGBMClassifier(**best_params, random_state=seed)
    
    final_model = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", final_clf)
    ])

    display_results(final_model, 'LGBM', x_train_subset, y_train, x_test_subset, y_test)

    return best_params


def hyperOpt_LGBM(X_subset, Y, preprocessor, scorer,  seed):

    x_train_subset, x_test_subset, y_train, y_test = train_test_split(X_subset, Y, test_size=0.2, random_state=seed)

    # Define the objective function
    def objective(params):
        #print(f"Trying params: {params}")
        model = XGBClassifier(**params, verbose=-1)
        score = cross_val_score(model, x_train_subset, y_train, cv=3, scoring=scorer).mean()
        return {'loss': -score, 'status': STATUS_OK}

    search_space = {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),  # ~0.0001 to 1
        'max_depth': hp.choice('max_depth', [2, 3, 4, 5, 6, 7, 8, 9]),  # Integer selection
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),  # Integer 50-300 in steps of 10
        'subsample': hp.uniform('subsample', 0.5, 1.0),  # Use 50-100% of data per round
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),  # Use 50-100% of features per tree
        'gamma': hp.loguniform('gamma', -5, 1),  # ~0.0067 to 2.71
        'reg_alpha': hp.loguniform('reg_alpha', -5, 1),  # L1 regularization ~0.0067 to 2.71
        'reg_lambda': hp.loguniform('reg_lambda', -5, 2),  # L2 regularization ~0.0067 to 7.39
    }

    # Run the optimization
    best_params = fmin(objective, search_space, algo=tpe.suggest, max_evals=500)

    # Print the best hyperparameters
    best_params = space_eval(search_space, best_params)
    print("Best hyperparameters:", best_params)

    # Evaluate the model with the best hyperparameters
    best_params['n_estimators'] = int(best_params['n_estimators'])

    final_clf = XGBClassifier(**best_params, random_state=seed)
    final_model = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", final_clf)
    ])
    final_model.fit(x_train_subset, y_train)
    accuracy = final_model.score(x_test_subset,y_test)
    print("Accuracy:", accuracy)