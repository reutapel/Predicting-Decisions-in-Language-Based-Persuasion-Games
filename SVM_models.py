from sklearn.svm import SVR, SVC
import numpy as np
import pandas as pd
import utils
import logging
from sklearn.dummy import DummyClassifier, DummyRegressor


class SVMTotal:
    def __init__(self, features, model_name, kernel: str=None, degree: int=None):
        if 'svm' in str.lower(model_name):
            self.model = SVR(gamma='scale', kernel=kernel, degree=degree)
        elif 'avg' in str.lower(model_name):
            self.model = DummyRegressor(strategy='mean')
        elif 'med' in str.lower(model_name):
            self.model = DummyRegressor(strategy='median')
        elif 'per_raisha_baseline' in str.lower(model_name):
            self.per_raisha = None
        else:
            logging.error('Model name not in: svm, average, median')
            print('Model name not in: svm, average, median')
            raise Exception('Model name not in: svm, average, median')
        self.features = features
        self.model_name = model_name

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        if 'per_raisha_baseline' in str.lower(self.model_name):
            train_y.name = 'labels'
            train_x = train_x.merge(train_y, right_index=True, left_index=True)
            self.per_raisha = pd.DataFrame(train_x.groupby(by='raisha').labels.mean())
            self.per_raisha.columns = ['predictions']
        else:
            train_x = train_x[self.features]
            self.model = self.model.fit(train_x, train_y)

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        if 'per_raisha_baseline' in str.lower(self.model_name):
            validation_x = validation_x.merge(self.per_raisha, left_on='raisha', right_index=True)
            validation_x.index = validation_x.sample_id
            predictions = validation_x.predictions
        else:
            validation_x = validation_x[self.features]
            predictions = self.model.predict(validation_x)
        validation_y.name = 'labels'
        predictions = pd.Series(predictions, index=validation_y.index, name='predictions')
        if predictions.dtype == float:  # regression- create bins to measure the F-score
            bin_prediction, bin_test_y = utils.create_bin_columns(predictions, validation_y)
            four_bin_prediction, four_bin_test_y = utils.create_4_bin_columns(predictions, validation_y)
        else:
            bin_prediction, bin_test_y = pd.Series(name='bin_prediction'), pd.Series(name='bin_label')
            four_bin_prediction, four_bin_test_y =\
                pd.Series(name='four_bin_prediction'), pd.Series(name='four_bin_label')

        predictions = pd.DataFrame(predictions).join(validation_y).join(bin_test_y).join(bin_prediction)
        predictions = predictions.join(four_bin_test_y).join(four_bin_prediction)

        return predictions


class SVMTurn:
    def __init__(self, features, model_name, kernel: str=None, degree: int=None):
        if 'svm' in str.lower(model_name):
            self.model = SVC(gamma='scale', kernel=kernel, degree=degree)
        elif 'ewg' in str.lower(model_name):
            self.model = DummyClassifier(strategy='stratified')
        elif 'mvc' in str.lower(model_name):
            self.model = DummyClassifier(strategy='most_frequent')
        else:
            logging.error('Model name not in: svm, stratified, most_frequent')
            print('Model name not in: svm, stratified, most_frequent')
            raise Exception('Model name not in: svm, stratified, most_frequent')
        self.features = features
        self.model_name = model_name

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        if 'svm' in str.lower(self.model_name):
            train_x.index = train_x.sample_id
            train_x_with_predictions = pd.DataFrame()
            if 'features' in train_x.columns and 'round_number' in train_x.columns and\
                    'raisha' in train_x.columns and 'prev_round_label' in self.features:
                rounds = train_x.round_number.unique()
                for round_num in rounds:
                    train_round = train_x.loc[train_x.round_number == round_num].copy(deep=True)
                    # work on rounds = first_round_saifa
                    first_round_saifa = train_round.loc[train_round.round_number == train_round.raisha + 1]
                    first_round_saifa_features = pd.DataFrame(first_round_saifa['features'].values.tolist(),
                                                              columns=self.features, index=first_round_saifa.sample_id)
                    train_round_y = train_y.loc[first_round_saifa.index]
                    self.model = self.model.fit(first_round_saifa_features, train_round_y)
                    predictions_first_round_saifa = self.model.predict(first_round_saifa_features)
                    predictions_first_round_saifa = pd.Series(predictions_first_round_saifa, name='prev_round_label',
                                                              index=first_round_saifa.sample_id)
                    predictions_first_round_saifa = train_round[['pair_id', 'raisha']].\
                        merge(predictions_first_round_saifa, left_index=True, right_index=True)
                    # change -1,1 predictions to be 0,1 features
                    predictions_first_round_saifa.prev_round_label = \
                        np.where(predictions_first_round_saifa.prev_round_label == -1, 0, 1)
                    # merge with the previous round prediction
                    # work on rounds > first_round_saifa
                    train_round = train_round.loc[train_round.round_number > train_round.raisha + 1]
                    if train_round.empty and round_num == 1:
                        predictions_pair_id = predictions_first_round_saifa
                        # change -1,1 predictions to be 0,1 features
                        train_x_with_predictions = pd.concat([train_x_with_predictions, first_round_saifa_features])
                        continue
                    train_round = train_round.merge(predictions_pair_id, on=['pair_id', 'raisha'], how='left').\
                        set_index(train_round.index)
                    train_round_features = pd.DataFrame(train_round['features'].values.tolist(),
                                                        columns=self.features, index=train_round.sample_id)
                    # remove prev_round_label and put the prediction instead
                    train_round_features = train_round_features.drop('prev_round_label', axis=1)
                    prev_round_prediction = train_round[['prev_round_label']].copy(deep=True)
                    train_round_features = train_round_features.merge(prev_round_prediction, left_index=True,
                                                                      right_index=True)
                    predictions = self.model.predict(train_round_features)
                    predictions = pd.Series(predictions, name='prev_round_label', index=train_round.sample_id)
                    predictions = train_round[['pair_id', 'raisha']].merge(predictions, left_index=True,
                                                                           right_index=True)
                    # change -1,1 predictions to be 0,1 features
                    predictions.prev_round_label = np.where(predictions.prev_round_label == -1, 0, 1)
                    predictions_pair_id = pd.concat([predictions_first_round_saifa, predictions])
                    train_x_with_predictions = pd.concat([train_x_with_predictions, first_round_saifa_features,
                                                          train_round_features])
                    # fit each time with all the rounds <= round_number
                    train_round_y = train_y.loc[train_x_with_predictions.index]
                    self.model = self.model.fit(train_x_with_predictions, train_round_y)
                # fit model after we have all predictions
                self.model = self.model.fit(train_x_with_predictions, train_y)

            else:
                logging.exception('No features or round_number or raisha column when running SVMTurn model '
                                  '--> can not run it')
                return
        elif 'ewg' in str.lower(self.model_name) or 'mvc' in str.lower(self.model_name):
            if 'features' in train_x.columns:
                train_x = pd.DataFrame(train_x['features'].values.tolist(), columns=self.features)
                self.model = self.model.fit(train_x, train_y)
            else:
                logging.exception('No features column when running SVMTurn model --> can not run it')
        else:
            logging.error('Model name not in: svm, ewg, mvc, avg, med')
            print('Model name not in: svm, ewg, mvc, avg, med')
            raise Exception('Model name not in: svm, ewg, mvc, avg, med')

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        if 'svm' in str.lower(self.model_name):
            all_predictions = pd.Series()
            predictions_pair_id = pd.DataFrame()
            validation_x.index = validation_x.sample_id
            if 'features' in validation_x.columns and 'round_number' in validation_x.columns and\
                    'raisha' in validation_x.columns and 'prev_round_label' in self.features:
                raisha_options = validation_x.raisha.unique()
                for raisha in raisha_options:
                    for round_num in range(raisha+1, 11):
                        validation_round = validation_x.loc[(validation_x.round_number == round_num) &
                                                            (validation_x.raisha == raisha)].copy(deep=True)
                        validation_round_features = pd.DataFrame(validation_round['features'].values.tolist(),
                                                                 columns=self.features, index=validation_round.sample_id)
                        if round_num > raisha + 1:  # the first round in the saifa --> no prediction for prev round
                            # merge with the previous round prediction
                            validation_round = validation_round.merge(predictions_pair_id, on=['pair_id', 'raisha']).\
                                set_index(validation_round.index)
                            # remove prev_round_label and put the prediction instead
                            validation_round_features = validation_round_features.drop('prev_round_label', axis=1)
                            prev_round_prediction = validation_round[['prev_round_label']].copy(deep=True)
                            validation_round_features = validation_round_features.merge(prev_round_prediction,
                                                                                        left_index=True, right_index=True)

                        predictions = self.model.predict(validation_round_features)
                        predictions = pd.Series(predictions, name='prev_round_label', index=validation_round.sample_id)
                        all_predictions = pd.concat([all_predictions, predictions])
                        predictions_pair_id = validation_round[['pair_id', 'raisha']].merge(predictions, left_index=True,
                                                                                            right_index=True)
                        # change -1,1 predictions to be 0,1 features
                        predictions_pair_id.prev_round_label =\
                            np.where(predictions_pair_id.prev_round_label == -1, 0, 1)

                predictions = validation_x[['raisha', 'pair_id', 'round_number']].\
                    join(pd.Series(all_predictions, name='predictions')). join(pd.Series(validation_y, name='labels'))
                return predictions

            else:
                logging.exception('No features or round_number or raisha column when running SVMTurn model '
                                  '--> can not run it')
                return
        elif 'ewg' in str.lower(self.model_name) or 'mvc' in str.lower(self.model_name):
            data = validation_x[validation_x.columns[1]].copy(deep=True)
            predictions = self.model.predict(data)
            validation_x.index = validation_x.sample_id
            predictions = validation_x[['raisha', 'pair_id', 'round_number']].\
                join(pd.Series(predictions, name='predictions', index=validation_x.index)).\
                join(pd.Series(validation_y, name='labels'))
            return predictions
        else:
            logging.error('Model name not in: ssvm, ewg, mvc, avg, med')
            print('Model name not in: svm, ewg, mvc, avg, med')
            raise Exception('Model name not in: svm, ewg, mvc, avg, med')
