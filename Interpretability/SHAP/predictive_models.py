import sklearn
import xgboost.sklearn
import catboost
from sklearn.dummy import DummyRegressor
import pandas as pd
import logging
import utils
import os

models = {'regression': {'svm': 'SVR',
                         'randomforest': 'RandomForestRegressor',
                         'xgboost': 'XGBRFRegressor',
                         'catboost': 'CatBoostRegressor'},
          'classification': {'svm': 'SVC',
                             'randomforest': 'RandomForestClassifier',
                             'xgboost': 'XGBClassifier',
                             'catboost': 'CatBoostClassifier'}}


class PredictiveModel:
    def __init__(self, features, model_name, hyper_parameters: dict, model_num: int, fold: int, fold_dir: str,
                 excel_models_results: pd.ExcelWriter, trained_model=None, model_type: str='regression'):
        self.features = features
        self.model_name = model_name
        self.model_num = model_num
        self.fold = fold
        self.fold_dir = fold_dir
        self.model_table_writer = pd.ExcelWriter(
            os.path.join(excel_models_results, f'Results_fold_{fold}_model_{model_num}.xlsx'), engine='xlsxwriter')
        self.trained_model = trained_model
        if self.trained_model is not None:
            self.model = trained_model

        else:
            if 'svm' in str.lower(model_name):
                self.model = getattr(sklearn.svm, models[model_type]['svm'])(
                    kernel=hyper_parameters['kernel'], degree=hyper_parameters['degree'])
            elif 'mean' in str.lower(model_name):
                self.model = DummyRegressor(strategy='mean')
            elif 'median' in str.lower(model_name):
                self.model = DummyRegressor(strategy='median')
            elif 'randomforest' in str.lower(model_name):
                self.model = getattr(sklearn.ensemble, models[model_type]['randomforest'])(
                    n_estimators=hyper_parameters['n_estimators'],
                    max_depth=hyper_parameters['max_depth'],
                    min_samples_split=hyper_parameters['min_samples_split'],
                    min_samples_leaf=hyper_parameters['min_samples_leaf'])
            elif 'xgboost' in str.lower(model_name):
                self.model = getattr(xgboost, models[model_type]['xgboost'])(
                    learning_rate=hyper_parameters['learning_rate'],
                    n_estimators=hyper_parameters['n_estimators'],
                    max_depth=hyper_parameters['max_depth'],
                    min_child_weight=hyper_parameters['min_child_weight'],
                    gamma=hyper_parameters['gamma'],
                    subsample=hyper_parameters['subsample'],
                    objective='reg:squarederror',)
            elif 'catboost' in str.lower(model_name):
                self.model = getattr(catboost, models[model_type]['catboost'])(
                    iterations=hyper_parameters['iterations'],
                    depth=hyper_parameters['depth'],
                    learning_rate=hyper_parameters['learning_rate'],
                    l2_leaf_reg=hyper_parameters['l2_leaf_reg'],
                    bootstrap_type='Bayesian',  # Poisson (supported for GPU only);Bayesian;Bernoulli;No
                    bagging_temperature=1,  # for Bayesian bootstrap_type; 1=exp;0=1
                    leaf_estimation_method='Newton',  # Gradient;Newton
                    leaf_estimation_iterations=2,
                    boosting_type='Ordered')  # Ordered-small data sets; Plain
            else:
                logging.error('Model name not in: CatBoost, lightgbm, XGBoost', 'RandomForest')
                print('Model name not in: CatBoost, lightgbm, XGBoost', 'RandomForest')
                raise Exception('Model name not in: CatBoost, lightgbm, XGBoost', 'RandomForest')

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        train_x = train_x[self.features]
        self.model = self.model.fit(train_x, train_y.values.ravel())

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        validation_x = validation_x[self.features]
        if 'xgboost' in str.lower(self.model_name):
            validation_x = validation_x[self.model._Booster.feature_names]
        predictions = self.model.predict(validation_x)
        validation_y.columns = ['labels']
        predictions = pd.Series(predictions, index=validation_y.index, name='predictions')
        predictions = pd.DataFrame(predictions).join(validation_y)
        utils.save_model_prediction(model_to_dave=self.model, model_name=self.model_name, data_to_save=predictions,
                                    fold_dir=self.fold_dir, fold=self.fold, model_num=self.model_num,
                                    table_writer=self.model_table_writer, save_model=True)

        return predictions
