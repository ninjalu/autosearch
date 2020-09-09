from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
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
                'n_estimators': [100, 500, 1000],
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
            'XGB': None
        }

        self.best_model = None
        self.predictions = None

    def fit(self, X, y):
        '''
        Fit X y train and validation to each ensemble model
        Save the best hyperparameters and evaluation metrics
        '''

        rf = RandomForestClassifier()
        rf_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=self.hyperP_search['RF'],
            scoring='f1_macro',
            n_iter=200, cv=5, verbose=2, random_state=42, n_jobs=-1)
        rf_search.fit(X, y)
        self.hyperP_best['RF'] = rf_search.best_params_
        self.metrics['RF'] = {'f1_macro': rf_search.best_score_}

        ab = AdaBoostClassifier()
        ab_search = RandomizedSearchCV(
            estimator=ab,
            param_distributions=self.hyperP_search['AB'],
            scoring='f1_macro',
            n_iter=200, cv=5, verbose=2, random_state=42, n_jobs=-1
        )
        ab_search.fit(X, y)
        self.hyperP_best['AB'] = ab_search.best_params_
        self.metrics['AB'] = {'f1_macro': ab_search.best_score_}

        xgb = XGBClassifier()
        xgb_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=self.hyperP_search['XGB'],
            scoring='f1_macro',
            n_iter=200, cv=5, verbose=2, random_state=42, n_jobs=-1
        )
        xgb_search.fit(X, y)
        self.hyperP_best['XGB'] = xgb_search.best_params_
        self.metrics['XGB'] = {'f1_macro': xgb_search.best_score_}

    def fit_predict_best_model(self, X, y, X_test):
        '''
        Find the best model
        Fit the whole training date with the best model
        '''
        if max(self.metrics) == 'RF':
            rf = RandomForestClassifier(
                n_estimators=self.hyperP_best['RF']['n_estimators'],
                min_samples_split=self.hyperP_best['RF']['min_samples_split'],
                min_samples_leaf=self.hyperP_best['RF']['min_samples_leaf'],
                max_depth=self.hyperP_best['RF']['max_depth'],
                bootstrap=self.hyperP_best['RF']['bootstrap'],
                n_jobs=-1,
                random_state=42
            )
            self.best_model = rf.fit(X, y)
            self.predictions = rf.predict(X_test)
            return self.predictions

        elif max(self.metrics) == 'AB':
            ab = AdaBoostClassifier(
                n_estimators=self.hyperP_best['AB']['n_estimators'],
                learning_rate=self.hyperP_best['AB']['learning_rate'],
                random_state=42
            )

            self.best_model = ab.fit(X, y)
            self.predictions = ab.predict(X_test)
            return self.predictions

        else:
            xgb = XGBClassifier(
                n_estimators=self.hyperP_best['XGB']['n_estimators'],
                subsample=self.hyperP_best['XGB']['subsample'],
                min_child_weight=self.hyperP_best['XGB']['min_child_weight'],
                max_depth=self.hyperP_best['XGB']['max_depth'],
                gamma=self.hyperP_best['XGB']['gamma'],
                eta=self.hyperP_best['XGB']['eta'],
                colsample_bytree=self.hyperP_best['XGB']['colsample_bytree']
            )

            self.best_model = xgb.fit(X, y)
            self.predictions = xgb.predict(X_test)
            return self.predictions

    def metrics_plot(self):
        '''
        Plot out the evaluation metrics for each model and the ensemble model
        '''
        pass
