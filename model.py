from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd


class Ensem_Model:
    ''' 
    This class ensembles different ensemble models
    Performs hyperparameter tuning for each model
    Evaluate model performances
    Combine ensembel models
    '''

    def __init__(self):
        '''
        Initiate training and validation sets
        Library containing hyperparameters for Random Forest, Ada boosting and XGBoost
        Library containing evaluation metrics/scores
        '''

        self.model_weights = {
            'RF': None,
            'AB': None,
            'XGB': None,
            'ORD': None
        }
        self.hyperP_best = {
            'RF': None,
            'AB': None,
            'XGB': None
        }

        self.hyperP_search = {
            'RF': {
                'n_estimators': [100, 300, 500, 800, 1200],
                'max_depth': [5, 8, 15, 25, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5, 10],
                'bootstrap': [True, False]
            },
            'AB': {
                'n_estimators': [100, 300, 500, 800, 1200],
                'learning_rate': [0.001, 0.01, 0.1, 0.2]
            },
            'XGB': {
                'n_estimators': [100, 1000],
                'max_depth': [1, 5, 10],
                'min_child_weight': [1, 5, 10],
                'subsample': [0.8, 1],
                'colsample_bytree': [0.8, 1],
                'eta': [0.001, 0.01, 0.1, 0.2],
                'gamma': [0.5, 1, 2, 5]
            }
        }

        self.metrics = {
            'RF': None,
            'AB': None,
            'XGB': None,
            'ORD': None
        }

    def fit(self, X, y):
        '''
        Fit X y train and validation to each ensemble model
        Save the best hyperparameters and evaluation metrics
        '''

        rf = RandomForestClassifier()
        rf_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=self.hyperP_search['RF'],
            scoring='roc_auc_ovr',
            n_iter=200, cv=5, verbose=2, random_state=42, n_jobs=-1)
        rf_search.fit(X, y)
        self.hyperP_best['RF'] = rf_search.best_params_
        self.metrics['RF'] = {'roc_auc_ovr': rf_search.best_score_}

        ab = AdaBoostClassifier()
        ab_search = RandomizedSearchCV(
            estimator=ab,
            param_distributions=self.hyperP_search['AB'],
            scoring='roc_auc_ovr',
            n_iter=200, cv=5, verbose=2, random_state=42, n_jobs=-1
        )
        ab_search.fit(X, y)
        self.hyperP_best['AB'] = ab_search.best_params_
        self.metrics['AB'] = {'roc_auc_ovr': ab_search.best_score_}

    def ensemble(self):
        '''
        Find the best combination of ensemble models
        Return their hyperparameters and evaluation metrics
        '''
        pass

    def metrics_plot(self):
        '''
        Plot out the evaluation metrics for each model and the ensemble model
        '''
        pass
