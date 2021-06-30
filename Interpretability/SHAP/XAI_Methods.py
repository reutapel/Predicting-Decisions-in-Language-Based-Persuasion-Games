import shap
import numpy as np
import pandas as pd
import logging


class XAIMethods:
    def __init__(self, model, X_test, X_train, XAI_method_name: str, model_type: str):
        self.model = model
        self.X_test = X_test
        self.X_train = X_train

        # self.y = y
        self.XAI_method_name = XAI_method_name
        self.model_type = model_type

        if 'shap' in str.lower(self.XAI_method_name):
            # explain the model's predictions using SHAP

            if 'svm' == str.lower(self.model_type):
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train, link="logit")
                self.shap_values = self.explainer.shap_values(X_test, nsamples=100)

            elif str.lower(self.model_type) in ['randomforest', 'xgboost', 'catboost']:
                self.explainer = shap.TreeExplainer(model)
                self.shap_values = self.explainer.shap_values(X_test)

        else:
            logging.error('XAI method name not included. this class supports: SHAP')
            print('XAI method name not included. this class supports: SHAP')
            raise Exception('XAI method name not included. this class supports: SHAP')


    '''https://github.com/slundberg/shap/issues/632'''
    def get_shap_feature_mean_values(self):
        vals = np.abs(self.shap_values).mean(0)

        feature_importance = pd.DataFrame(list(zip(self.X_test.columns, vals)), columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        return feature_importance

    def visualize_shap(self):
        shap.summary_plot(self.shap_values, self.X_test)
        shap.summary_plot(self.shap_values, self.X_test, plot_type="bar")
