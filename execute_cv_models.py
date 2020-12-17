import pandas as pd
import os
import argparse
import numpy as np
import utils
import SVM_models
import models
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from dataset_readers import LSTMDatasetReader, TransformerDatasetReader
import torch.optim as optim
from allennlp.training.metrics import *
from datetime import datetime
from allennlp.data.vocabulary import Vocabulary
import logging
import joblib
import torch
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from collections import defaultdict
import copy
from allennlp.nn.regularizers import RegularizerApplicator, L1Regularizer
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm

per_round_predictions_name = 'per_round_predictions'
per_round_labels_name = 'per_round_labels'


class ExecuteEvalModel:
    """
    This is a father class for all models we want to compare.
    Load data, split to train-validation based on the fold_split_dict.
    Train model on train data.
    Predict on validation data.
    Evaluate model on validation data.
    Save model, log, return the evaluation.
    """
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str, excel_models_results: str,
                 hyper_parameters_dict: dict, trained_model=None, trained_model_dir=None, model_file_name=None,
                 test_data_file_name: str=None, predict_type: str='validation'):
        """

        :param model_num:
        :param fold:
        :param fold_dir:
        :param model_type:
        :param model_name:
        :param data_file_name:
        :param fold_split_dict:
        :param table_writer:
        :param data_directory:
        :param excel_models_results:
        :param hyper_parameters_dict:
        :param trained_model:
        :param trained_model_dir:
        :param model_file_name:
        :param test_data_file_name:
        :param predict_type: need to be validation or test
        """

        self.model_num = model_num
        self.fold = fold
        self.fold_dir = fold_dir
        self.model_type = model_type
        self.model_name = model_name
        self.data_file_name = data_file_name
        self.test_data_file_name = test_data_file_name
        self.fold_split_dict = fold_split_dict
        self.table_writer = table_writer
        self.model = None
        self.prediction = pd.DataFrame()
        self.data_directory = data_directory
        self.train_pair_ids = self.fold_split_dict['train']
        self.val_pair_ids = self.fold_split_dict['validation']
        self.test_pair_ids = self.fold_split_dict['test']
        self.trained_model = trained_model
        self.trained_model_dir = trained_model_dir
        self.model_file_name = model_file_name
        self.predict_type = predict_type
        self.model_table_writer = pd.ExcelWriter(
            os.path.join(excel_models_results, f'Results_fold_{fold}_model_{model_num}.xlsx'), engine='xlsxwriter')
        print(f'Create Model: model num: {model_num},\nmodel_type: {model_type},\nmodel_name: {model_name}. '
              f'\nData file name: {data_file_name}'
              f'\nModel hyper parameters: {hyper_parameters_dict}')
        logging.info(f'Create Model: model num: {model_num}, model_type: {model_type}, model_name: {model_name}. '
                     f'Data file name: {data_file_name}')

    def load_data_create_model(self):
        """This function should load the data, split to train-validation and create the model"""
        raise NotImplementedError

    def fit_validation(self):
        """This function should fit the model on the train data, predict on the validation data and dave the results"""
        raise NotImplementedError

    def eval_model(self):
        """This function should use the prediction of the model and eval these results"""

        raise NotImplementedError

    def predict(self):
        """This function use trained model to predict the labels of the test set"""
        raise NotImplementedError

    def save_model_prediction(self, data_to_save: pd.DataFrame, save_model=True, sheet_prefix_name: str='All',
                              save_fold=None, element_to_save: dict=None):
        """
        Save the model predictions and the model itself
        :param data_to_save: the data to save
        :param save_model: whether to save the model
        :param sheet_prefix_name: the sheet prefix name to save
        :param save_fold: the fold we want to save the model in
        :param element_to_save: if we want to save something that is not the model itself: {element_name: element}
        :return:
        """
        if save_fold is None:
            save_fold = self.fold_dir
        # save the model
        if save_model:
            logging.info(f'Save model {self.model_num}_{self.model_type}_{self.model_name}_fold_{self.fold}.pkl')
            joblib.dump(self.model, os.path.join(
                save_fold, f'{self.model_num}_{self.model_type}_{self.model_name}_fold_{self.fold}.pkl'))
        if element_to_save is not None:
            for element_name, element in element_to_save.items():
                joblib.dump(element, os.path.join(save_fold, f'{self.model_num}_{element_name}_fold_{self.fold}.pkl'))

        for table_writer in [self.table_writer, self.model_table_writer]:
            utils.write_to_excel(
                table_writer, f'Model_{self.model_num}_{sheet_prefix_name}_fold_{self.fold}',
                headers=[f'{sheet_prefix_name} predictions for model {self.model_num}: {self.model_name} of type '
                         f'{self.model_type} in fold {self.fold}'], data=data_to_save)

    def total_payoff_calculate_measures(self, final_total_payoff_prediction_column: str, total_payoff_label_column: str,
                                        raisha_column_name: str = 'raisha', prediction_df: pd.DataFrame=None,
                                        bin_label: pd.Series = None, bin_predictions: pd.Series = None,
                                        four_bin_label: pd.Series = None, four_bin_predictions: pd.Series = None,
                                        prediction_type: str=''):
        """
        Calculate the measures for seq models per raisha
        :param total_payoff_label_column: the name of the label column
        :param final_total_payoff_prediction_column: the name of the prediction label
        :param raisha_column_name:
        :param prediction_df: if we don't want to use self.prediction
        :param prediction_type: if we want to use seq and reg predictions- so we have a different column for each.
        :param bin_label: the 3 bin label
        :param bin_predictions: the 3 bin prediction
        :param four_bin_label: the 4 bin label
        :param four_bin_predictions: the 4 bin prediction
        :return:
        """

        if prediction_df is None:
            prediction_df = self.prediction
        # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
        if bin_label is not None:
            only_bin_label = copy.deepcopy(bin_label['bin_label'])
        else:
            only_bin_label = bin_label
        if bin_predictions is not None:
            only_bin_predictions = copy.deepcopy(bin_predictions['bin_predictions'])
        else:
            only_bin_predictions = bin_predictions
        if four_bin_label is not None:
            only_four_bin_label = copy.deepcopy(four_bin_label['four_bin_label'])
        else:
            only_four_bin_label = four_bin_label
        if four_bin_predictions is not None:
            only_four_bin_predictions = copy.deepcopy(four_bin_predictions['four_bin_predictions'])
        else:
            only_four_bin_predictions = four_bin_predictions
        _, results_dict = utils.calculate_measures_for_continues_labels(
                prediction_df, final_total_payoff_prediction_column=final_total_payoff_prediction_column,
                total_payoff_label_column=total_payoff_label_column,
                bin_label=only_bin_label, bin_predictions=only_bin_predictions,
                label_options=['total future payoff < 1/3', '1/3 < total future payoff < 2/3',
                               'total future payoff > 2/3'], prediction_type=prediction_type)
        _, results_dict_four_bins = utils.calculate_measures_for_continues_labels(
                prediction_df, final_total_payoff_prediction_column=final_total_payoff_prediction_column,
                total_payoff_label_column=total_payoff_label_column, already_calculated=True,
                bin_label=only_four_bin_label, bin_predictions=only_four_bin_predictions,
                bin_label_column_name='four_bin_label', bin_prediction_column_name='four_bin_predictions',
                label_options=['total future payoff < 1/4', '1/4 < total future payoff < 1/2',
                               '1/2 < total future payoff < 3/4',
                               'total future payoff > 3/4'], prediction_type=prediction_type)
        results_dict = utils.update_default_dict(results_dict, results_dict_four_bins)
        if raisha_column_name in prediction_df.columns:  # do the raisha analysis
            raisha_options = prediction_df[raisha_column_name].unique()
            all_raisha_dict = defaultdict(dict)
            for raisha in raisha_options:
                raisha_data = prediction_df.loc[prediction_df[raisha_column_name] == raisha]
                if bin_label is not None:
                    raisha_bin_label = bin_label.loc[bin_label[raisha_column_name] == raisha]['bin_label']
                else:
                    raisha_bin_label = bin_label
                if bin_predictions is not None:
                    raisha_bin_predictions =\
                        bin_predictions.loc[bin_predictions[raisha_column_name] == raisha]['bin_predictions']
                else:
                    raisha_bin_predictions = bin_predictions

                if four_bin_label is not None:
                    raisha_four_bin_label = four_bin_label.loc[four_bin_label[raisha_column_name] == raisha][
                        'four_bin_label']
                else:
                    raisha_four_bin_label = four_bin_label
                if four_bin_predictions is not None:
                    raisha_four_bin_predictions = \
                        four_bin_predictions.loc[four_bin_predictions[raisha_column_name] == raisha][
                            'four_bin_predictions']
                else:
                    raisha_four_bin_predictions = four_bin_predictions
                _, results_dict_raisha = utils.calculate_measures_for_continues_labels(
                    raisha_data, final_total_payoff_prediction_column=final_total_payoff_prediction_column,
                    total_payoff_label_column=total_payoff_label_column,
                    bin_label=raisha_bin_label, bin_predictions=raisha_bin_predictions,
                    label_options=['total future payoff < 1/3', '1/3 < total future payoff < 2/3',
                                   'total future payoff > 2/3'], raisha=f'raisha_{str(int(raisha))}',
                    prediction_type=prediction_type)
                _, four_results_dict_raisha = utils.calculate_measures_for_continues_labels(
                    raisha_data, final_total_payoff_prediction_column=final_total_payoff_prediction_column,
                    total_payoff_label_column=total_payoff_label_column,
                    bin_label=raisha_four_bin_label, bin_predictions=raisha_four_bin_predictions,
                    label_options=['total future payoff < 1/4', '1/4 < total future payoff < 1/2',
                                   '1/2 < total future payoff < 3/4', 'total future payoff > 3/4'],
                    bin_label_column_name='four_bin_label', bin_prediction_column_name='four_bin_predictions',
                    raisha=f'raisha_{str(int(raisha))}', already_calculated=True, prediction_type=prediction_type)
                all_raisha_dict.update(results_dict_raisha)
                all_raisha_dict = utils.update_default_dict(all_raisha_dict, four_results_dict_raisha)

            results_dict = utils.update_default_dict(results_dict, all_raisha_dict)

        return results_dict

    def flat_seq_predictions_multiple_columns(self, label_column_name_per_round: str,
                                              prediction_column_name_per_round: str) -> pd.DataFrame:
        """
        Use the prediction DF to get one column of all rounds predictions and labels, in order to calculate
        the per round measures
        :param label_column_name_per_round: the name od the label column per round (for example: y_0, y_1, ..., y_9)
        :param prediction_column_name_per_round: the name od the prediction column per round
        (for example: y_prime_0, y_prime_1, ..., y_prime_9)
        :return: pd.Dataframe with 2 columns: labels and predictions with the labels and predictions per round for
        the saifa data
        """

        flat_data = dict()
        index_to_use = None
        for flat_col_name, name_for_dict in [[label_column_name_per_round, per_round_labels_name],
                                             [prediction_column_name_per_round, per_round_predictions_name]]:
            cols = [f'{flat_col_name}_{i}' for i in range(10)]
            data = self.prediction[cols]
            data.index = self.prediction.raisha
            stack_data = data.stack()
            # get only the relevant rounds --> the saifa rounds
            stack_data = stack_data.loc[stack_data != '-']
            if flat_col_name == label_column_name_per_round:
                # rounds: 1-10
                temp_index = [f'raisha_{r}_round_{int(ro.split("_")[1])+1}' for r, ro in stack_data.index.values]
            stack_data = stack_data.reset_index(drop=True)
            stack_data = stack_data.astype(int)
            flat_data[name_for_dict] = stack_data

        try:
            if (flat_data[per_round_labels_name].index == flat_data[per_round_predictions_name].index).sum() == \
                    flat_data[per_round_predictions_name].shape[0]:  # if there are index that are not the same
                flat_index = pd.DataFrame(temp_index, index=index_to_use)[0].str.split('_', expand=True)
                if flat_index.shape[1] == 4:
                    flat_index.columns = [1, 'raisha', 2, 'round_number']
                else:
                    logging.exception(f'flat index in flat_seq_predictions_multiple_columns does not have 4 columns')
                    raise Exception(f'flat index in flat_seq_predictions_multiple_columns does not have 4 columns')
                flat_index = flat_index[['raisha', 'round_number']]
                flat_data_df = pd.DataFrame.from_dict(flat_data)
                flat_data_df = flat_data_df.merge(flat_index, left_index=True, right_index=True)
                # save the flat data
                self.save_model_prediction(data_to_save=flat_data_df, save_model=False, sheet_prefix_name='per_round')

                return flat_data_df

        except Exception:
            logging.exception(f'Failed in flat_seq_predictions_multiple_columns: index after flat are not the same')
            return pd.DataFrame()

    def flat_seq_predictions_list_column(self, label_column_name_per_round: str,
                                         prediction_column_name_per_round: str) -> pd.DataFrame:
        """
        Use the prediction DF to get one column of all rounds predictions and labels, in order to calculate
        the per round measures
        :param label_column_name_per_round: the name od the label column per round (for example: y_0, y_1, ..., y_9)
        :param prediction_column_name_per_round: the name od the prediction column per round
        (for example: y_prime_0, y_prime_1, ..., y_prime_9)
        :return: pd.Dataframe with 2 columns: labels and predictions with the labels and predictions per round for
        the saifa data
        """

        flat_data_dict = dict()
        for list_column, new_column in [[label_column_name_per_round, per_round_labels_name],
                                        [prediction_column_name_per_round, per_round_predictions_name]]:
            # create a pd with [new_column, 'raisha', 'sample_id'] columns
            flat_data = copy.deepcopy(self.prediction)
            # reset index to get numeric index for the future merge
            flat_data['sample_id'] = flat_data.index
            flat_data.reset_index(inplace=True, drop=True)
            flat_data = flat_data[[list_column, 'raisha', 'sample_id']]
            lens_of_lists = flat_data[list_column].apply(len)
            origin_rows = range(flat_data.shape[0])
            destination_rows = np.repeat(origin_rows, lens_of_lists)
            non_list_cols = [idx for idx, col in enumerate(flat_data.columns) if col != list_column]
            expanded_df = flat_data.iloc[destination_rows, non_list_cols].copy()
            expanded_df[new_column] = [i for items in flat_data[list_column] for i in items]
            # remove non 0/1 rows and reset index
            expanded_df = expanded_df.loc[expanded_df[new_column].isin(['0', '1'])]
            # create round number column
            round_number = pd.Series()
            for index, round_num in lens_of_lists.iteritems():
                round_number =\
                    round_number.append(pd.Series(list(range(11-round_num, 11)), index=np.repeat(index, round_num)))
            expanded_df['round_number'] = round_number
            expanded_df.reset_index(inplace=True, drop=True)
            flat_data_dict[new_column] = expanded_df[[new_column]]
            flat_data_dict['metadata'] = expanded_df[['raisha', 'sample_id', 'round_number']]

        # concat the new labels and new predictions per round
        flat_data = flat_data_dict[per_round_labels_name].join(flat_data_dict[per_round_predictions_name]).\
            join(flat_data_dict['metadata'])
        flat_data.reset_index(inplace=True, drop=True)
        # save flat data
        self.save_model_prediction(data_to_save=flat_data, save_model=False, sheet_prefix_name='round')

        return flat_data


class ExecuteEvalSVM(ExecuteEvalModel):
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str, hyper_parameters_dict: dict,
                 excel_models_results: str, trained_model=None, trained_model_dir=None, model_file_name=None,
                 test_data_file_name: str=None, predict_type: str='validation'):
        super(ExecuteEvalSVM, self).__init__(model_num, fold, fold_dir, model_type, model_name, data_file_name,
                                             fold_split_dict, table_writer, data_directory, excel_models_results,
                                             hyper_parameters_dict, trained_model, trained_model_dir, model_file_name,
                                             test_data_file_name=test_data_file_name, predict_type=predict_type)
        self.label_name = hyper_parameters_dict['label_name']
        self.train_x, self.train_y, self.validation_x, self.validation_y, self.test_y, self.test_x =\
            None, None, None, None, None, None
        self.features_file_name = 'features_' + self.data_file_name.split('all_data_', 1)[1].split('.pkl')[0]+'.xlsx'
        print(f'Features file name: {self.features_file_name}')

        if 'kernel' in hyper_parameters_dict.keys():
            self.kernel = hyper_parameters_dict['kernel']
        else:
            self.kernel = 'rbf'

        self.degree = 0
        if 'degree' in hyper_parameters_dict.keys() and type(hyper_parameters_dict['degree']) == int:
            self.degree = hyper_parameters_dict['degree']

    def load_data_create_model(self):
        print(f'Load and create Data file name: {self.data_file_name}')
        logging.info(f'Load and create Data file name: {self.data_file_name}')
        print(f'Load and create Test Data file name: {self.test_data_file_name}')
        logging.info(f'Load and create Test Data file name: {self.test_data_file_name}')
        if 'pkl' in self.data_file_name:
            data = joblib.load(os.path.join(self.data_directory, self.data_file_name))
            if self.test_data_file_name is not None:
                test_data = joblib.load(os.path.join(self.data_directory, self.test_data_file_name))
        else:
            data = pd.read_csv(os.path.join(self.data_directory, self.data_file_name))
            if self.test_data_file_name is not None:
                test_data = joblib.load(os.path.join(self.data_directory, self.test_data_file_name))

        # get the feature columns
        x_columns = data.columns.tolist()
        x_columns.remove(self.label_name)
        data.index = data.sample_id
        # get train data
        train_data = data.loc[data.pair_id.isin(self.train_pair_ids)]
        self.train_y = train_data[self.label_name]
        self.train_x = train_data[x_columns]
        # get validation data
        validation_data = data.loc[data.pair_id.isin(self.val_pair_ids)]
        self.validation_y = validation_data[self.label_name]
        self.validation_x = validation_data[x_columns]
        # get test data
        if self.test_data_file_name is not None:
            if 'sample_id' in test_data.columns:
                test_data.index = test_data.sample_id
            test_data = test_data.loc[test_data.pair_id.isin(self.test_pair_ids)]
            self.test_y = test_data[self.label_name]
            self.test_x = test_data[x_columns]

        # load features file:
        features = pd.read_excel(os.path.join(self.data_directory, self.features_file_name))
        features = features[0].tolist()

        # create or load model
        if self.trained_model is not None:
            self.model = self.trained_model
        else:
            self.model = getattr(SVM_models, self.model_type)(features, self.model_name, self.kernel, self.degree)

    def fit_validation(self):
        print(f'fit and predict model {self.model_name}')
        logging.info(f'fit and predict model {self.model_name}')
        self.model.fit(self.train_x, self.train_y)
        self.predict()

    def predict(self):
        if self.predict_type == 'validation':
            x = self.validation_x
            y = self.validation_y
            save_model = True
        elif self.predict_type == 'test':
            if self.test_data_file_name is not None:
                x = self.test_x
                y = self.test_y
                save_model = False
            else:
                print("If predict_type is test, test_data_file_name can't be None")
                return
        else:
            print(f'predict_type must be validation or test, {self.predict_type} was passed')
            return
        self.prediction = self.model.predict(x, y)
        self.save_model_prediction(data_to_save=self.prediction, save_model=save_model)

    def eval_model(self):
        if self.predict_type == 'validation':
            x = self.validation_x
        elif self.predict_type == 'test':
            x = self.test_x
        else:
            print(f'predict_type must be validation or test, {self.predict_type} was passed')
            return
        print(f'Eval model {self.model_name}')
        logging.info(f'Eval model {self.model_name}')
        if self.model_type == 'SVMTotal' or 'proportion' in self.model_name:
            if 'raisha' in x.columns and 'proportion' not in self.model_name:  # only relevant for svm
                self.prediction = self.prediction.join(x.raisha)
            try:
                if 'labels' in self.prediction.columns and 'predictions' in self.prediction.columns:
                    # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
                    results_dict = self.total_payoff_calculate_measures(
                        final_total_payoff_prediction_column='predictions', total_payoff_label_column='labels')

                    return results_dict

            except Exception:
                logging.exception(f'labels or predictions not in SVMTotal prediction DF')
                return

        elif self.model_type == 'SVMTurn' or 'per_round' in self.model_name:
            # change -1 to 1 in labels and predictions
            self.prediction.labels = np.where(self.prediction.labels == 1, 1, 0)
            self.prediction.predictions = np.where(self.prediction.predictions == 1, 1, 0)

            # measures per round
            results_dict = utils.per_round_analysis(
                self.prediction, predictions_column='predictions', label_column='labels',
                label_options=['DM chose stay home', 'DM chose hotel'], function_to_run='calculate_per_round_measures')
            if 'raisha' in self.prediction:
                results_dict_per_round_per_raisha = utils.per_round_analysis(
                    self.prediction, predictions_column='predictions', label_column='labels',
                    label_options=['DM chose stay home', 'DM chose hotel'],
                    function_to_run='calculate_per_round_per_raisha_measures')
                results_dict = utils.update_default_dict(results_dict, results_dict_per_round_per_raisha)

            # create the total payoff label and calculate calculate_measures_for_continues_labels
            pairs = self.prediction.pair_id.unique()
            prediction_total_payoff = self.prediction.copy(deep=True)
            total_payoff_label_predictions = defaultdict(dict)
            for pair_id in pairs:
                pair_data = prediction_total_payoff.loc[prediction_total_payoff.pair_id == pair_id]
                raisha_options = pair_data.raisha.unique()
                for raisha in raisha_options:
                    pair_raisha_data = pair_data.loc[pair_data.raisha == raisha].copy(deep=True)
                    total_payoff_prediction = np.average(pair_raisha_data.predictions.values)
                    total_payoff_label = np.average(pair_raisha_data.labels.values)
                    total_payoff_label_predictions[f'{pair_id}_{raisha}']['raisha'] = raisha
                    total_payoff_label_predictions[f'{pair_id}_{raisha}']['total_payoff_prediction'] =\
                        total_payoff_prediction
                    total_payoff_label_predictions[f'{pair_id}_{raisha}']['total_payoff_label'] =\
                        total_payoff_label

            total_payoff_label_predictions = pd.DataFrame.from_dict(total_payoff_label_predictions).T

            # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
            bin_prediction, bin_test_y = utils.create_bin_columns(
                total_payoff_label_predictions.total_payoff_prediction,
                total_payoff_label_predictions.total_payoff_label)
            total_payoff_label_predictions = total_payoff_label_predictions.join(bin_test_y).join(bin_prediction)
            # four bin analysis
            four_bin_prediction, four_bin_test_y = utils.create_4_bin_columns(
                total_payoff_label_predictions.total_payoff_prediction,
                total_payoff_label_predictions.total_payoff_label)
            total_payoff_label_predictions =\
                total_payoff_label_predictions.join(four_bin_test_y).join(four_bin_prediction)
            results_dict_total_future_payoff = self.total_payoff_calculate_measures(
                final_total_payoff_prediction_column='total_payoff_prediction',
                total_payoff_label_column='total_payoff_label', prediction_df=total_payoff_label_predictions)
            results_dict = utils.update_default_dict(results_dict, results_dict_total_future_payoff)

            return results_dict

        else:
            logging.exception(f'Model type is no SVMTurn and no SVMTotal')
            return


class ExecuteEvalLSTM(ExecuteEvalModel):
    def __init__(self, model_num: int, fold: int, fold_dir: str, model_type: str, model_name: str, data_file_name: str,
                 fold_split_dict: dict, table_writer: pd.ExcelWriter, data_directory: str, hyper_parameters_dict: dict,
                 excel_models_results: str, trained_model=None, trained_model_dir=None, model_file_name=None,
                 test_data_file_name: str=None, predict_type: str='validation'):
        super(ExecuteEvalLSTM, self).__init__(model_num, fold, fold_dir, model_type, model_name, data_file_name,
                                              fold_split_dict, table_writer, data_directory, excel_models_results,
                                              hyper_parameters_dict, trained_model, trained_model_dir, model_file_name,
                                              test_data_file_name=test_data_file_name, predict_type=predict_type)
        if 'lstm_hidden_dim' in hyper_parameters_dict.keys():
            self.lstm_hidden_dim = int(hyper_parameters_dict['lstm_hidden_dim'])
        else:
            if 'LSTM' in model_type:
                logging.exception(f'LSTM model type must have lstm_hidden_dim in its hyper_parameters_dict')
                print(f'LSTM model type must have lstm_hidden_dim in its hyper_parameters_dict')
                raise Exception(f'LSTM model type must have lstm_hidden_dim in its hyper_parameters_dict')
            else:
                self.lstm_hidden_dim = 0
        if 'num_epochs' in hyper_parameters_dict.keys():
            self.num_epochs = int(hyper_parameters_dict['num_epochs'])
        else:
            self.num_epochs = 100
        if 'batch_size' in hyper_parameters_dict.keys():
            self.batch_size = int(hyper_parameters_dict['batch_size'])
        else:
            self.batch_size = 10
        if 'features_max_size' in hyper_parameters_dict.keys():
            self.features_max_size = int(hyper_parameters_dict['features_max_size'])
        else:
            self.features_max_size = 0
        if 'raisha_num_features' in hyper_parameters_dict.keys():
            self.raisha_num_features = int(hyper_parameters_dict['raisha_num_features'])
        else:
            self.raisha_num_features = 0
        if 'num_encoder_layers' in hyper_parameters_dict.keys():
            self.num_encoder_layers = int(hyper_parameters_dict['num_encoder_layers'])
        else:
            self.num_encoder_layers = 6
        if 'num_decoder_layers' in hyper_parameters_dict.keys():
            self.num_decoder_layers = int(hyper_parameters_dict['num_decoder_layers'])
        else:
            self.num_decoder_layers = self.num_encoder_layers
        if 'positional_encoding' in hyper_parameters_dict.keys():
            self.positional_encoding = hyper_parameters_dict['positional_encoding']
        else:
            self.positional_encoding = 'sinusoidal'
        if 'linear_dropout' in hyper_parameters_dict.keys():
            if hyper_parameters_dict['linear_dropout'] is not None:
                self.linear_dropout = float(hyper_parameters_dict['linear_dropout'])
            else:
                self.linear_dropout = None
        else:
            self.linear_dropout = None

        self.activation = 'relu'

        self.all_validation_accuracy = list()
        self.all_train_accuracy = list()
        self.all_seq_predictions = pd.DataFrame()
        self.all_reg_predictions = pd.DataFrame()
        self.trainer = None
        self.linear_hidden_dim = None
        self.avg_loss = 1.0  # if we don't use 2 losses - the weight of each of them should be 1
        self.turn_loss = 1.0
        self.avg_turn_loss = 1.0
        self.hotel_label_0 = None
        self.vocab = None
        self.cuda_device = None
        self.run_log_directory = utils.set_folder(datetime.now().strftime(
            f'{self.model_num}_{self.model_type}_{self.model_name}_{self.num_epochs}_epochs_fold_num_{self.fold}'),
            self.fold_dir)
        try:
            if 'turn' in model_type:
                self.predict_seq = True
            else:
                self.predict_seq = False

            if 'avg' in model_type:
                self.predict_avg_total_payoff = True
            else:
                self.predict_avg_total_payoff = False

            if 'last_hidden' in model_type:
                self.use_last_hidden_vec = True
            else:
                self.use_last_hidden_vec = False

            if 'use_transformer' in model_type:
                self.use_transformer_encoder = True
            else:
                self.use_transformer_encoder = False

            if 'use_raisha_attention' in model_type:
                self.use_raisha_attention = True
            else:
                self.use_raisha_attention = False

            if 'use_raisha_LSTM' in model_type:
                self.use_raisha_LSTM = True
            else:
                self.use_raisha_LSTM = False

            if 'only_raisha' in model_type:
                self.only_raisha = True
            else:
                self.only_raisha = False

            if 'avg_loss' in hyper_parameters_dict.keys():
                self.avg_loss = float(hyper_parameters_dict['avg_loss'])
            if 'turn_loss' in hyper_parameters_dict.keys():
                self.turn_loss = float(hyper_parameters_dict['turn_loss'])
            if 'avg_turn_loss' in hyper_parameters_dict.keys():
                self.avg_turn_loss = float(hyper_parameters_dict['avg_turn_loss'])
            if 'linear_hidden_dim' in hyper_parameters_dict.keys():
                self.linear_hidden_dim = int(hyper_parameters_dict['linear_hidden_dim'])
            if 'lstm_dropout' in hyper_parameters_dict.keys():
                self.lstm_dropout = float(hyper_parameters_dict['lstm_dropout'])
            else:
                self.lstm_dropout = 0.0
            if 'num_layers' in hyper_parameters_dict.keys():
                self.num_layers = int(hyper_parameters_dict['num_layers'])
            else:
                self.num_layers = 1
            if 'feedforward_hidden_dim_prod' in hyper_parameters_dict.keys():
                self.feedforward_hidden_dim_prod = float(hyper_parameters_dict['feedforward_hidden_dim_prod'])
            else:
                self.feedforward_hidden_dim_prod = 4

            if 'BiLSTM' in hyper_parameters_dict.keys():
                self.BiLSTM = True
            else:
                self.BiLSTM = False

            if 'lr' in hyper_parameters_dict.keys():
                self.lr = float(hyper_parameters_dict['lr'])
            else:
                self.lr = 1e-3

        except Exception:
            logging.exception(f'None of the optional types were given --> can not continue')
            return

    def load_data_create_model(self):
        print(f'Load and create Data file name: {self.data_file_name}')
        logging.info(f'Load and create Data file name: {self.data_file_name}')
        all_data_file_path = os.path.join(self.data_directory, self.data_file_name)

        # load train data
        if 'LSTM' in self.model_type or 'Attention' in self.model_type:
            train_reader = LSTMDatasetReader(pair_ids=self.train_pair_ids,
                                             use_transformer=self.use_transformer_encoder,
                                             use_raisha_attention=self.use_raisha_attention,
                                             use_raisha_LSTM=self.use_raisha_LSTM,
                                             raisha_num_features=self.raisha_num_features)
            validation_reader = LSTMDatasetReader(pair_ids=self.val_pair_ids,
                                                  use_transformer=self.use_transformer_encoder,
                                                  use_raisha_attention=self.use_raisha_attention,
                                                  use_raisha_LSTM=self.use_raisha_LSTM,
                                                  raisha_num_features=self.raisha_num_features)

        elif 'Transformer' in self.model_type:
            train_reader = TransformerDatasetReader(pair_ids=self.train_pair_ids, only_raisha=self.only_raisha,
                                                    features_max_size=self.features_max_size)
            validation_reader = TransformerDatasetReader(pair_ids=self.val_pair_ids, only_raisha=self.only_raisha,
                                                         features_max_size=self.features_max_size)
        else:
            logging.exception(f'Model type should include LSTM or Transformer to use this class')
            print(f'Model type should include LSTM or Transformer to use this class')
            raise Exception(f'Model type should include LSTM or Transformer to use this class')

        train_instances = train_reader.read(all_data_file_path)
        validation_instances = validation_reader.read(all_data_file_path)
        self.vocab = Vocabulary.from_instances(train_instances + validation_instances)

        self.hotel_label_0 = True if self.vocab._index_to_token['labels'][0] == 'hotel' else False

        metrics_dict_seq = {
            'Accuracy': CategoricalAccuracy(),
            'F1measure_hotel_label': F1Measure(positive_label=self.vocab._token_to_index['labels']['hotel']),
            'F1measure_home_label': F1Measure(positive_label=self.vocab._token_to_index['labels']['stay_home']),
        }

        metrics_dict_reg = {
            'mean_absolute_error': MeanAbsoluteError(),
        }

        # batch_size should be: 10 or 9 depends on the input
        # and not shuffle so all the data of the same pair will be in the same batch
        iterator = BasicIterator(batch_size=self.batch_size)  # , instances_per_epoch=10)
        iterator.index_with(self.vocab)
        if 'LSTM' in self.model_type:
            if self.use_transformer_encoder:
                encoder_layer = TransformerEncoderLayer(
                    train_reader.input_dim, nhead=8, dropout=self.lstm_dropout,
                    dim_feedforward=int(self.feedforward_hidden_dim_prod * train_reader.input_dim))
                encoder_norm = LayerNorm(train_reader.input_dim)
                lstm = TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm)
            else:
                if self.use_raisha_LSTM:
                    LSTM_input_size = train_reader.raisha_num_features
                else:
                    LSTM_input_size = train_reader.num_features
                lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(LSTM_input_size, self.lstm_hidden_dim,
                                                           batch_first=True, num_layers=self.num_layers,
                                                           dropout=self.lstm_dropout, bidirectional=self.BiLSTM))
            self.model = models.LSTMBasedModel(
                encoder=lstm, metrics_dict_seq=metrics_dict_seq, metrics_dict_reg=metrics_dict_reg, vocab=self.vocab,
                predict_seq=self.predict_seq, predict_avg_total_payoff=self.predict_avg_total_payoff,
                linear_dim=self.linear_hidden_dim, seq_weight_loss=self.turn_loss,
                reg_weight_loss=self.avg_loss, dropout=self.linear_dropout, reg_seq_weight_loss=self.avg_turn_loss,
                use_last_hidden_vec=self.use_last_hidden_vec, use_transformer_encode=self.use_transformer_encoder,
                input_dim=train_reader.input_dim, use_raisha_attention=self.use_raisha_attention,
                raisha_num_features=train_reader.raisha_num_features, use_raisha_LSTM=self.use_raisha_LSTM,
                linear_layers_activation=self.activation)
        elif 'Transformer' in self.model_type:
            if 'turn_linear' in self.model_type:
                self.linear_hidden_dim = int(0.5 * train_reader.input_dim)
            self.model = models.TransformerBasedModel(
                vocab=self.vocab, metrics_dict_seq=metrics_dict_seq, metrics_dict_reg=metrics_dict_reg,
                predict_avg_total_payoff=self.predict_avg_total_payoff, linear_dim=self.linear_hidden_dim,
                batch_size=self.batch_size, input_dim=train_reader.input_dim,  # reg_seq_weight_loss=self.avg_turn_loss,
                feedforward_hidden_dim=int(self.feedforward_hidden_dim_prod * train_reader.input_dim),
                num_decoder_layers=self.num_decoder_layers, num_encoder_layers=self.num_encoder_layers,
                seq_weight_loss=self.turn_loss, reg_weight_loss=self.avg_loss, transformer_dropout=self.lstm_dropout,
                positional_encoding=self.positional_encoding, dropout=self.linear_dropout, only_raisha=self.only_raisha,
                linear_layers_activation=self.activation, predict_seq=self.predict_seq
            )
        else:
            logging.exception(f'Model type should include LSTM or Transformer or Attention to use this class')
            print(f'Model type should include LSTM or Transformer or Attention to use this class')
            raise Exception(f'Model type should include LSTM or Transformer or Attention to use this class')

        print(self.model)
        if torch.cuda.is_available():
            self.cuda_device = 0
            print('Cuda is available')
            logging.info('Cuda is available')

            torch.backends.cudnn.benchmark = True

            self.model = self.model.cuda()

        else:
            self.cuda_device = -1
            print('Cuda is not available')
            logging.info('Cuda is not available')

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        validation_metric = '+Accuracy' if self.predict_seq else '-loss'

        self.trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_instances,
            validation_dataset=validation_instances,
            num_epochs=self.num_epochs,
            shuffle=False,
            serialization_dir=self.run_log_directory,
            patience=10,
            histogram_interval=10,
            cuda_device=self.cuda_device,
            validation_metric=validation_metric,
        )

    def fit_validation(self):
        print(f'fit and predict model {self.model_name}')
        logging.info(f'fit and predict model {self.model_name}')
        model_dict = self.trainer.train()
        self.organize_predictions(model_dict)

        # delete all items in the folder, except of the final model, the logs, the metrics and the bets model
        print('Delete all items in the folder in ExecuteEvalLSTM')
        for file in os.listdir(self.run_log_directory):
            if 'metrics_epoch' in file or 'model_state_epoch' in file or 'training_state_epoch' in file:
                os.remove(os.path.join(self.run_log_directory, file))

    def organize_predictions(self, model_dict: dict=None):
        if model_dict is None:
            model_dict = {'best_epoch': 0}
            save_model = False
            element_to_save = None
        else:
            element_to_save = {'model_dict': model_dict}
            save_model = True
        if self.predict_seq:
            self.all_seq_predictions = pd.DataFrame.from_dict(self.model.seq_predictions, orient='index')
            # select the  best epoch if exists
            if f"predictions_{model_dict['best_epoch']}" in self.all_seq_predictions:
                self.all_seq_predictions['final_prediction'] =\
                    self.all_seq_predictions[f"predictions_{model_dict['best_epoch']}"]
            else:
                max_predict_column = max([int(column.split('_')[1]) for column in self.all_seq_predictions.columns if
                                          'predictions' in column])
                self.all_seq_predictions['final_prediction'] =\
                    self.all_seq_predictions[f'predictions_{max_predict_column}']
            # select the  best epoch if exists
            if f"total_payoff_prediction_{model_dict['best_epoch']}" in self.all_seq_predictions:
                self.all_seq_predictions['final_total_payoff_prediction'] = \
                    self.all_seq_predictions[f"total_payoff_prediction_{model_dict['best_epoch']}"]
            else:
                max_total_payoff_predict_column = max([int(column.split('_')[3]) for
                                                       column in self.all_seq_predictions.columns if
                                                       'total_payoff_prediction_' in column])
                self.all_seq_predictions['final_total_payoff_prediction'] =\
                    self.all_seq_predictions[f'total_payoff_prediction_{max_total_payoff_predict_column}']
            self.save_model_prediction(data_to_save=self.all_seq_predictions, sheet_prefix_name='seq',
                                       save_model=save_model, save_fold=self.run_log_directory,
                                       element_to_save=element_to_save)
            self.all_seq_predictions = self.all_seq_predictions[['is_train', 'labels', 'total_payoff_label', 'raisha',
                                                                 'final_prediction', 'final_total_payoff_prediction']]
            self.all_seq_predictions = self.all_seq_predictions.loc[self.all_seq_predictions.is_train==False]
            self.prediction = self.all_seq_predictions

        if self.predict_avg_total_payoff:
            self.all_reg_predictions = self.model.reg_predictions
            # select the  best epoch if exists
            if f"prediction_{model_dict['best_epoch']}" in self.all_reg_predictions:
                self.all_reg_predictions['final_total_payoff_prediction'] = \
                    self.all_reg_predictions[f"prediction_{model_dict['best_epoch']}"]
            else:
                max_predict_column = max([int(column.split('_')[1]) for column in self.all_reg_predictions.columns if
                                          'prediction_' in column])
                self.all_reg_predictions['final_total_payoff_prediction'] =\
                    self.all_reg_predictions[f'prediction_{max_predict_column}']
            self.save_model_prediction(data_to_save=self.all_reg_predictions, sheet_prefix_name='reg',
                                       save_fold=self.run_log_directory, element_to_save=element_to_save,
                                       save_model=save_model)
            self.all_reg_predictions = self.all_reg_predictions[['is_train', 'sample_id', 'total_payoff_label',
                                                                 'raisha', 'final_total_payoff_prediction']]
            self.all_reg_predictions = self.all_reg_predictions.loc[self.all_reg_predictions.is_train==False]
            self.prediction = self.all_reg_predictions

    def eval_model(self):
        print(f'Eval model {self.model_name}')
        logging.info(f'Eval model {self.model_name}')
        results_dict = list()
        if 'total_payoff_label' in self.prediction.columns and\
                'final_total_payoff_prediction' in self.prediction.columns:
            # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
            if self.predict_avg_total_payoff and self.predict_seq:  # avg_turn models - need to be both
                seq_pred_type = '_seq'
            else:  # only seq/reg prediction --> no need to split measures
                seq_pred_type = ''
            for df, prediction_type in [[self.all_reg_predictions, ''], [self.all_seq_predictions, seq_pred_type]]:
                if not df.empty:
                    self.prediction = df
                    bin_prediction, bin_test_y = utils.create_bin_columns(
                        self.prediction.final_total_payoff_prediction, self.prediction.total_payoff_label,
                        hotel_label_0=self.hotel_label_0)
                    bin_prediction = self.prediction[['raisha']].join(bin_prediction)
                    bin_test_y = self.prediction[['raisha']].join(bin_test_y)
                    four_bin_prediction, four_bin_test_y = utils.create_4_bin_columns(
                        self.prediction.final_total_payoff_prediction, self.prediction.total_payoff_label,
                        hotel_label_0=self.hotel_label_0)
                    four_bin_prediction = self.prediction[['raisha']].join(four_bin_prediction)
                    four_bin_test_y = self.prediction[['raisha']].join(four_bin_test_y)

                    # this function return mae, rmse, mse and bin analysis: prediction, recall, fbeta
                    results_dict.append(self.total_payoff_calculate_measures(
                        prediction_type=prediction_type,
                        final_total_payoff_prediction_column='final_total_payoff_prediction',
                        total_payoff_label_column='total_payoff_label',
                        bin_label=bin_test_y, bin_predictions=bin_prediction,
                        four_bin_label=four_bin_test_y, four_bin_predictions=four_bin_prediction))
            if len(results_dict) > 1:  # the loop happened more than once
                results_dict = utils.update_default_dict(results_dict)
            else:
                results_dict = results_dict[0]

            # measures per round
            if self.predict_seq:  # and not self.predict_avg_total_payoff:
                self.prediction = self.all_seq_predictions
                flat_seq_predictions = self.flat_seq_predictions_list_column(
                    label_column_name_per_round='labels',
                    prediction_column_name_per_round='final_prediction')
                label_options = ['DM chose hotel', 'DM chose stay home'] if self.hotel_label_0\
                    else ['DM chose stay home', 'DM chose hotel']
                results_dict_per_round = utils.per_round_analysis(
                    flat_seq_predictions, predictions_column=per_round_predictions_name,
                    label_column=per_round_labels_name, label_options=label_options,
                    function_to_run='calculate_per_round_measures')
                results_dict = utils.update_default_dict(results_dict, results_dict_per_round)

                if 'raisha' in flat_seq_predictions:
                    results_dict_per_round_per_raisha = utils.per_round_analysis(
                        flat_seq_predictions, predictions_column=per_round_predictions_name,
                        label_column=per_round_labels_name, label_options=label_options,
                        function_to_run='calculate_per_round_per_raisha_measures')
                    results_dict = utils.update_default_dict(results_dict, results_dict_per_round_per_raisha)

            return results_dict

        else:
            logging.exception(f'Error in eval model {self.model_name}')
            raise Exception(f'Error in eval model {self.model_name}')

    def predict(self):
        if self.trained_model is None:
            print('Need to pass trained model for this models')
            return
        # load test data
        all_data_file_path = os.path.join(self.data_directory, self.test_data_file_name)
        # load train data
        if 'LSTM' in self.model_type or 'Attention' in self.model_type:
            test_reader = LSTMDatasetReader(pair_ids=self.test_pair_ids,
                                            use_transformer=self.use_transformer_encoder,
                                            use_raisha_attention=self.use_raisha_attention,
                                            use_raisha_LSTM=self.use_raisha_LSTM,
                                            raisha_num_features=self.raisha_num_features)

        elif 'Transformer' in self.model_type:
            test_reader = TransformerDatasetReader(pair_ids=self.test_pair_ids, only_raisha=self.only_raisha,
                                                   features_max_size=self.features_max_size)
        else:
            logging.exception(f'Model type should include LSTM or Transformer to use this class')
            print(f'Model type should include LSTM or Transformer to use this class')
            raise Exception(f'Model type should include LSTM or Transformer to use this class')

        test_instances = test_reader.read(all_data_file_path)
        iterator = BasicIterator(batch_size=self.batch_size)  # , instances_per_epoch=10)
        iterator.index_with(self.vocab)

        # reload the model.
        self.model.load_state_dict(self.trained_model.state_dict())
        predictor = models.Predictor(self.model, iterator=iterator, cuda_device=self.cuda_device)
        predictor.predict(ds=test_instances)
        self.organize_predictions()
