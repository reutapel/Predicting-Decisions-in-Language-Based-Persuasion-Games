import joblib
import math
from typing import Union
from typing import *
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import math
import os
import logging
import copy
from collections import defaultdict

base_directory = os.path.abspath(os.curdir)

per_round_predictions_name = 'per_round_predictions'
per_round_labels_name = 'per_round_labels'


def update_default_dict(orig_dict: defaultdict, dict2: defaultdict=None, dict_list: list=None):
    """This function get an orig defaultdict and 1 defaultdicts and merge them ot a list of defaultdicts -->
     one of them have to be passed"""

    if dict2 is not None:
        dicts = [dict2]
    elif dict_list is not None:
        dicts = dict_list
    elif type(orig_dict) == list:
        dicts = orig_dict[1:]
        orig_dict = orig_dict[0]
    else:
        print('Both dict2 and dict_list are None --> can not continue')
        return orig_dict
    for my_dict in dicts:
        if my_dict is not None:
            for k, v in my_dict.items():
                if k in orig_dict.keys():
                    orig_dict[k].update(v)
                else:
                    orig_dict[k] = v

    return orig_dict


def calculate_measures(train_data: pd.DataFrame, validation_data: pd.DataFrame, metric_list: List[str],
                       label_col_name: str='label') -> tuple([dict, dict]):
    """
    This function get train and validation data that has label and prediction columns and calculate the measures in
    the metric_list
    :param train_data: pd.DataFrame with the train data, has to have at least label and prediction columns
    :param validation_data: pd.DataFrame with the validation data, has to have at least label and prediction columns
    :param metric_list: a list with the metric names to calculate
    :param label_col_name: the name of the label column
    :return:
    """
    # calculate metric(y_true, y_pred)
    validation_metric_dict = dict()
    train_metric_dict = dict()
    for metric in metric_list:
        validation_metric_dict[metric] =\
            getattr(metrics, metric)(validation_data[label_col_name], validation_data.prediction)
        train_metric_dict[metric] = getattr(metrics, metric)(train_data[label_col_name], train_data.prediction)

    return train_metric_dict, validation_metric_dict


def create_bin_columns(predictions: pd.Series, validation_y: pd.Series, hotel_label_0: bool=False):
    """
    Create the bin analysis column
    :param predictions: the continues prediction column
    :param validation_y: the continues label column
    :param hotel_label_0: if the label of the hotel option is 0
    :return:
    """

    # bin measures,
    # class: hotel_label == 1: predictions < 0.33 --> 0, 0.33<predictions<0.67 --> 1, predictions > 0.67 --> 2
    #        hotel_label == 0: predictions < 0.33 --> 2, 0.33<predictions<0.67 --> 1, predictions > 0.67 --> 0
    low_entry_rate_class = 2 if hotel_label_0 else 0
    high_entry_rate_class = 0 if hotel_label_0 else 2
    # for prediction
    keep_mask = predictions < 0.33
    bin_prediction = np.where(predictions < 0.67, 1, high_entry_rate_class)
    bin_prediction[keep_mask] = low_entry_rate_class
    bin_prediction = pd.Series(bin_prediction, name='bin_predictions', index=validation_y.index)
    # for test_y
    keep_mask = validation_y < 0.33
    bin_test_y = np.where(validation_y < 0.67, 1, high_entry_rate_class)
    bin_test_y[keep_mask] = low_entry_rate_class
    bin_test_y = pd.Series(bin_test_y, name='bin_label', index=validation_y.index)

    return bin_prediction, bin_test_y


def create_4_bin_columns(predictions: pd.Series, validation_y: pd.Series, hotel_label_0: bool=False):
    """
    Create the bin analysis column
    :param predictions: the continues prediction column
    :param validation_y: the continues label column
    :param hotel_label_0: if the label of the hotel option is 0
    :return:
    """

    # bin measures,
    # class: hotel_label == 1: predictions < 0.25 --> 0, 0.25<predictions<0.5 --> 1, 0.5<predictions<0.75 --> 2,
    #  predictions > 0.75 --> 3
    #        hotel_label == 0: predictions < 0.25 --> 3, 0.25<predictions<0.5 --> 2, 0.5<predictions<0.75 --> 1,
    #  predictions > 0.75 --> 0
    low_entry_rate_class = 3 if hotel_label_0 else 0
    med_1_entry_rate_class = 2 if hotel_label_0 else 1
    med_2_entry_rate_class = 1 if hotel_label_0 else 2
    high_entry_rate_class = 0 if hotel_label_0 else 3
    # for prediction
    med_1_mask = predictions.between(0.25, 0.5)
    med_2_mask = predictions.between(0.5, 0.75)
    high_mask = predictions.between(0.75, 2)
    bin_prediction = np.where(predictions < 0.25, low_entry_rate_class, high_entry_rate_class)
    bin_prediction[med_1_mask] = med_1_entry_rate_class
    bin_prediction[med_2_mask] = med_2_entry_rate_class
    bin_prediction[high_mask] = high_entry_rate_class
    bin_prediction = pd.Series(bin_prediction, name='four_bin_predictions', index=validation_y.index)

    # for test_y
    med_1_mask = validation_y.between(0.25, 0.5)
    med_2_mask = validation_y.between(0.5, 0.75)
    high_mask = validation_y.between(0.75, 2)
    bin_test_y = np.where(validation_y < 0.25, low_entry_rate_class, high_entry_rate_class)
    bin_test_y[med_1_mask] = med_1_entry_rate_class
    bin_test_y[med_2_mask] = med_2_entry_rate_class
    bin_test_y[high_mask] = high_entry_rate_class
    bin_test_y = pd.Series(bin_test_y, name='four_bin_label', index=validation_y.index)

    return bin_prediction, bin_test_y


def per_round_analysis(all_predictions: pd.DataFrame, predictions_column: str, label_column: str, label_options: list,
                       function_to_run):
    """
    Analyze per round results: calculate measures for all rounds and per round
    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :param function_to_run: the function to run: calculate_per_round_per_raisha_measures or calculate_per_round_measures
    :return:
    """
    results_dict = globals()[function_to_run](all_predictions, predictions_column, label_column, label_options)

    if 'round_number' in all_predictions.columns:  # analyze the results per round
        for current_round_number in all_predictions.round_number.unique():
            data = all_predictions.loc[all_predictions.round_number == current_round_number].copy(deep=True)
            results = globals()[function_to_run](data, predictions_column, label_column, label_options,
                                                 round_number=f'round_{int(current_round_number)}')
            results_dict = update_default_dict(results_dict, results)

    return results_dict


def calculate_per_round_per_raisha_measures(all_predictions: pd.DataFrame, predictions_column: str, label_column: str,
                                            label_options: list, round_number: str='All_rounds'):
    """

    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :param round_number: if we analyze specific round number
    :return:
    """
    raishas = all_predictions.raisha.unique()
    results_dict = defaultdict(dict)
    for raisha in raishas:
        data = copy.deepcopy(all_predictions.loc[all_predictions.raisha == raisha])
        results = calculate_per_round_measures(data, predictions_column, label_column, label_options,
                                               raisha=f'raisha_{int(raisha)}', round_number=round_number)
        results_dict.update(results)

    return results_dict


def calculate_per_round_measures(all_predictions: pd.DataFrame, predictions_column: str, label_column: str,
                                 label_options: list, raisha: str='All_raishas', round_number: str='All_rounds'):
    """

    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :param raisha: the suffix for the columns in raisha analysis
    :param round_number: if we analyze specific round number
    :return:
    """
    results_dict = defaultdict(dict)
    dict_key = f'{raisha} {round_number}'
    precision, recall, fbeta_score, support =\
        metrics.precision_recall_fscore_support(all_predictions[label_column], all_predictions[predictions_column])
    accuracy = metrics.accuracy_score(all_predictions[label_column], all_predictions[predictions_column])
    precision_macro, recall_macro, fbeta_score_macro, support_macro =\
        metrics.precision_recall_fscore_support(all_predictions[label_column], all_predictions[predictions_column],
                                                average='macro')

    # number of DM chose stay home
    final_labels = list(range(len(support)))
    # get the labels in the all_predictions DF
    true_labels = all_predictions[label_column].unique()
    true_labels.sort()
    for label_index, label in enumerate(true_labels):
        status_size = all_predictions[label_column].where(all_predictions[label_column] == label).dropna().shape[0]
        if status_size in support:
            index_in_support = np.where(support == status_size)[0][0]
            final_labels[index_in_support] = label_options[label_index]

    # create the results to return
    for measure, measure_name in [[precision, 'precision'], [recall, 'recall'], [fbeta_score, 'Fbeta_score']]:
        for i, label in enumerate(final_labels):
            results_dict[dict_key][f'Per_round_{measure_name}_{label}'] = round(measure[i]*100, 2)
    for measure, measure_name in [[precision_macro, 'precision_macro'], [recall_macro, 'recall_macro'],
                                  [fbeta_score_macro, 'fbeta_score_macro'], [accuracy, 'accuracy']]:
        results_dict[dict_key][f'Per_round_{measure_name}'] = round(measure*100, 2)

    return results_dict


def calculate_measures_for_continues_labels(all_predictions: pd.DataFrame, final_total_payoff_prediction_column: str,
                                            total_payoff_label_column: str, label_options: list,
                                            raisha: str = 'All_raishas', round_number: str = 'All_rounds',
                                            bin_label: pd.Series=None, bin_predictions: pd.Series=None,
                                            already_calculated: bool=False,
                                            bin_label_column_name: str='bin_label',
                                            bin_prediction_column_name: str='bin_predictions',
                                            prediction_type: str='') -> (pd.DataFrame, dict):
    """
    Calc and print the regression measures, including bin analysis
    :param all_predictions:
    :param total_payoff_label_column: the name of the label column
    :param final_total_payoff_prediction_column: the name of the prediction label
    :param label_options: list of the label option names
    :param raisha: if we run a raisha analysis this is the raisha we worked with
    :param round_number: for per round analysis
    :param bin_label: the bin label series, the index is the same as the total_payoff_label_column index
    :param bin_predictions: the bin predictions series, the index is the same as the total_payoff_label_column index
    :param prediction_type: if we want to use seq and reg predictions- so we have a different column for each.
    :param already_calculated: if we already calculated the measures, need to calculate again only the bin measures
    :param bin_label_column_name: the name of the bin label column if it is in the all_prediction df
    :param bin_prediction_column_name: the name of the bin prediction column if it is in the all_prediction df
    :return:
    """
    dict_key = f'{raisha} {round_number}'
    if 'is_train' in all_predictions.columns:
        data = all_predictions.loc[all_predictions.is_train == False]
    else:
        data = all_predictions

    results_dict = defaultdict(dict)
    predictions = data[final_total_payoff_prediction_column]
    gold_labels = data[total_payoff_label_column]
    mse = metrics.mean_squared_error(predictions, gold_labels)
    rmse = round(100 * math.sqrt(mse), 2)
    mae = round(100 * metrics.mean_absolute_error(predictions, gold_labels), 2)
    mse = round(100 * mse, 2)

    # calculate bin measures
    if bin_label_column_name and bin_prediction_column_name in all_predictions.columns:
        bin_label = all_predictions[bin_label_column_name]
        bin_predictions = all_predictions[bin_prediction_column_name]
    elif bin_label is None and bin_predictions is None:
        print(f'No bin labels and bin predictions')
        logging.info(f'No bin labels and bin predictions')
        raise Exception

    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(bin_label, bin_predictions)
    num_bins = len(label_options)
    precision_micro, recall_micro, fbeta_score_micro, support_micro =\
        metrics.precision_recall_fscore_support(bin_label, bin_predictions, average='micro')
    precision_macro, recall_macro, fbeta_score_macro, support_macro =\
        metrics.precision_recall_fscore_support(bin_label, bin_predictions, average='macro')

    # number of DM chose stay home
    final_labels = list(range(len(support)))
    for my_bin in range(len(label_options)):
        status_size = bin_label.where(bin_label == my_bin).dropna().shape[0]
        if status_size in support:
            index_in_support = np.where(support == status_size)[0]
            if final_labels[index_in_support[0]] in label_options and index_in_support.shape[0] > 1:
                # 2 bins with the same size --> already assign
                index_in_support = index_in_support[1]
            else:
                index_in_support = index_in_support[0]
            final_labels[index_in_support] = label_options[my_bin]

    for item in final_labels:
        if item not in label_options:  # status_size = 0
            final_labels.remove(item)

    accuracy = metrics.accuracy_score(bin_label, bin_predictions)
    results_dict[dict_key][f'Bin_{num_bins}_bins_Accuracy{prediction_type}'] = round(accuracy * 100, 2)

    # create the results to return
    for measure, measure_name in [[precision, 'precision'], [recall, 'recall'], [fbeta_score, 'Fbeta_score']]:
        for i in range(len(measure)):
            if f'Bin_{measure_name}_{final_labels[i]}{prediction_type}' in ['Bin_Fbeta_score_1', 'Bin_Fbeta_score_2',
                                                                            'Bin_Fbeta_score_3', 'Bin_precision_1',
                                                                            'Bin_precision_2', 'Bin_precision_3',
                                                                            'Bin_recall_1', 'Bin_recall_2',
                                                                            'Bin_recall_3']:
                print(f'Error: final_labels: {final_labels}, label_options: {label_options},'
                      f'already_calculated: {already_calculated}, raisha: {raisha}, rounds: {round_number}')
            results_dict[dict_key][f'Bin_{measure_name}_{final_labels[i]}{prediction_type}'] = round(measure[i]*100, 2)
    for measure, measure_name in [[precision_micro, 'precision_micro'], [recall_micro, 'recall_micro'],
                                  [fbeta_score_micro, 'Fbeta_score_micro'], [precision_macro, 'precision_macro'],
                                  [recall_macro, 'recall_macro'], [fbeta_score_macro, 'Fbeta_score_macro']]:
        results_dict[dict_key][f'Bin_{num_bins}_bins_{measure_name}{prediction_type}'] = round(measure * 100, 2)

    if not already_calculated:
        results_dict[dict_key][f'MSE{prediction_type}'] = mse
        results_dict[dict_key][f'RMSE{prediction_type}'] = rmse
        results_dict[dict_key][f'MAE{prediction_type}'] = mae

    results_pd = pd.DataFrame.from_dict(results_dict, orient='index')

    return results_pd, results_dict


def write_to_excel(table_writer: pd.ExcelWriter, sheet_name: str, headers: list, data: pd.DataFrame):
    """
    This function get header and data and write to excel
    :param table_writer: the ExcelWrite object
    :param sheet_name: the sheet name to write to
    :param headers: the header of the sheet
    :param data: the data to write
    :return:
    """
    if table_writer is None:
        return
    workbook = table_writer.book
    if sheet_name not in table_writer.sheets:
        worksheet = workbook.add_worksheet(sheet_name)
    else:
        worksheet = workbook.get_worksheet_by_name(sheet_name)
    table_writer.sheets[sheet_name] = worksheet

    data.to_excel(table_writer, sheet_name=sheet_name, startrow=len(headers), startcol=0)
    all_format = workbook.add_format({
        'valign': 'top',
        'border': 1})
    worksheet.set_column(0, data.shape[1], None, all_format)

    # headers format
    merge_format = workbook.add_format({
        'bold': True,
        'border': 2,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True,
    })
    for i, header in enumerate(headers):
        worksheet.merge_range(first_row=i, first_col=0, last_row=i, last_col=data.shape[1], data=header,
                              cell_format=merge_format)

    return


def set_folder(folder_name: str, father_folder_name: str = None, father_folder_path=None):
    """
    This function create new folder for results if does not exists
    :param folder_name: the name of the folder to create
    :param father_folder_name: the father name of the new folder
    :param father_folder_path: if pass the father folder path and not name
    :return: the new path or the father path if folder name is None
    """
    # create the father folder if not exists
    if father_folder_name is not None:
        path = os.path.join(base_directory, father_folder_name)
    else:
        path = father_folder_path
    if not os.path.exists(path):
        os.makedirs(path)
    # create the folder
    if folder_name is not None:
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)

    return path


def flat_seq_predictions_list_column(label_column_name_per_round: str,
                                     prediction_column_name_per_round: str,
                                     prediction: pd.DataFrame) -> pd.DataFrame:
    """
    Use the prediction DF to get one column of all rounds predictions and labels, in order to calculate
    the per round measures
    :param label_column_name_per_round: the name od the label column per round (for example: y_0, y_1, ..., y_9)
    :param prediction_column_name_per_round: the name od the prediction column per round
    (for example: y_prime_0, y_prime_1, ..., y_prime_9)
    :param prediction: the data
    :return: pd.Dataframe with 2 columns: labels and predictions with the labels and predictions per round for
    the saifa data
    """

    flat_data_dict = dict()
    for list_column, new_column in [[label_column_name_per_round, per_round_labels_name],
                                    [prediction_column_name_per_round, per_round_predictions_name]]:
        # create a pd with [new_column, 'raisha', 'sample_id'] columns
        flat_data = copy.deepcopy(prediction)
        # reset index to get numeric index for the future merge
        flat_data['sample_id'] = flat_data.index
        flat_data.reset_index(inplace=True, drop=True)
        flat_data = flat_data[[list_column, 'raisha', 'sample_id']]
        flat_data[list_column] =\
            flat_data[list_column].apply(lambda row: [int(item) for item in list(row) if item in ['0', '1']])
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

    return flat_data


def save_model_prediction(model_to_dave, model_name: str, data_to_save: pd.DataFrame, fold_dir: str, fold: int,
                          model_num: int, table_writer, save_model: bool=True,):
    """
    Save the model predictions and the model itself
    :param data_to_save: the data to save
    :param save_model: whether to save the model
    :param fold_dir: the fold we want to save the model in
    :return:
    """

    # save the model
    if save_model:
        logging.info(f'Save model {model_num}: {model_name}_fold_{fold}.pkl')
        joblib.dump(model_to_dave, os.path.join(
            fold_dir, f'{model_num}_{model_name}_fold_{fold}.pkl'))

    write_to_excel(
        table_writer, f'Model_{model_num}_{model_name}_fold_{fold}',
        headers=[f'Predictions for model {model_num} {model_name} in fold {fold}'], data=data_to_save)


def load_data(data_path: str, label_name: str, features_families: list,  test_pair_ids: list, train_pair_ids: list=None,
              id_column: str='pair_id', features_to_remove: Union[list, str]=None):
    """
    Load data from data_path and return: train_x, train_y, test_x, test_y
    :param data_path: path to data
    :param label_name: the label column name
    :param features_families: the families of features to use
    :param train_pair_ids: the pair ids for train data, if None- return only test data
    :param test_pair_ids: the pair ids for test data
    :param id_column: the name of the ID columns
    :param features_to_remove: list of features we don't want to use
    :return:
    """

    if 'pkl' in data_path:
        data = joblib.load(data_path)
    else:
        data = pd.read_csv(data_path)

    if train_pair_ids is not None:
        if id_column in data.columns:
            train_data = data.loc[data[id_column].isin(train_pair_ids)]
            train_data.index = train_data[id_column]
        elif 'meta_data' in data.columns and id_column in data.meta_data.columns:
            train_data = data.loc[data.meta_data[id_column].isin(train_pair_ids)]
            train_data.index = train_data.meta_data[id_column]
        else:
            print(f'meta_data and {id_column} not in data.columns')
            raise ValueError
        train_y = train_data[label_name]
        train_x = train_data[features_families]
        train_x.columns = train_x.columns.get_level_values(1)
        if features_to_remove is not None:
            if type(features_to_remove) == str:
                train_x_columns = [column for column in train_x.columns if features_to_remove not in column]
            else:
                train_x_columns = [column for column in train_x.columns if column not in features_to_remove]
            train_x = train_x[train_x_columns]
    else:
        train_y = None
        train_x = None

    if id_column in data.columns:
        test_data = data.loc[data[id_column].isin(test_pair_ids)]
        test_data.index = test_data[id_column]
    elif 'meta_data' in data.columns and id_column in data.meta_data.columns:
        test_data = data.loc[data.meta_data[id_column].isin(test_pair_ids)]
        test_data.index = test_data.meta_data[id_column]
    else:
        print(f'meta_data and {id_column} not in data.columns')
        raise ValueError
    test_y = test_data[label_name]
    test_x = test_data[features_families]
    test_x.columns = test_x.columns.get_level_values(1)

    if features_to_remove is not None:
        if type(features_to_remove) == str:
            test_x_columns = [column for column in test_x.columns if features_to_remove not in column]
        else:
            test_x_columns = [column for column in test_x.columns if column not in features_to_remove]
        test_x = test_x[test_x_columns]

    return train_x, train_y, test_x, test_y


def calculate_predictive_model_measures(all_predictions: pd.DataFrame, predictions_column: str='predictions',
                                        label_column: str='labels',
                                        label_options: list=['DM chose stay home', 'DM chose hotel']):
    """

    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :return:
    """
    results_dict = defaultdict(dict)
    precision, recall, fbeta_score, support =\
        metrics.precision_recall_fscore_support(all_predictions[label_column], all_predictions[predictions_column])
    accuracy = metrics.accuracy_score(all_predictions[label_column], all_predictions[predictions_column])
    precision_micro, recall_micro, fbeta_score_micro, support_micro =\
        metrics.precision_recall_fscore_support(all_predictions[label_column], all_predictions[predictions_column],
                                                average='micro')
    precision_macro, recall_macro, fbeta_score_macro, support_macro =\
        metrics.precision_recall_fscore_support(all_predictions[label_column], all_predictions[predictions_column],
                                                average='macro')

    # number of DM chose stay home
    final_labels = list(range(len(support)))
    # get the labels in the all_predictions DF
    true_labels = all_predictions[label_column].unique()
    true_labels.sort()
    for label_index, label in enumerate(true_labels):
        status_size = all_predictions[label_column].where(all_predictions[label_column] == label).dropna().shape[0]
        if status_size in support:
            index_in_support = np.where(support == status_size)[0][0]
            final_labels[index_in_support] = label_options[label_index]

    # create the results to return
    for measure, measure_name in [[precision, 'precision'], [recall, 'recall'], [fbeta_score, 'Fbeta_score']]:
        for i, label in enumerate(final_labels):
            results_dict[f'{measure_name}_{label}'] = round(measure[i]*100, 2)

    for measure, measure_name in [[precision_micro, 'precision_micro'], [recall_micro, 'recall_micro'],
                                  [fbeta_score_micro, 'Fbeta_score_micro'], [precision_macro, 'precision_macro'],
                                  [recall_macro, 'recall_macro'], [fbeta_score_macro, 'Fbeta_score_macro']]:
        results_dict[f'{measure_name}'] = round(measure * 100, 2)
    results_dict[f'Accuracy'] = round(accuracy * 100, 2)

    return results_dict


def calculate_continues_predictive_model_measures(all_predictions: pd.DataFrame, predictions_column: str='predictions',
                                                  label_column: str='labels'):
    """

    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :return:
    """
    results_dict = defaultdict(dict)
    predictions = all_predictions[predictions_column]
    gold_labels = all_predictions[label_column]
    mse = metrics.mean_squared_error(predictions, gold_labels)
    rmse = math.sqrt(mse)
    mae = metrics.mean_absolute_error(predictions, gold_labels)
    mse = mse

    # create the results to return
    for measure, measure_name in [[rmse, 'RMSE'], [mae, 'MAE'], [mse, 'MSE']]:
        results_dict[f'{measure_name}'] = round(measure * 100, 2)

    return results_dict


def flat_seq_predictions_list_column(prediction: pd.DataFrame, label_column_name_per_round: str,
                                     prediction_column_name_per_round: str) -> pd.DataFrame:
    """
    Use the prediction DF to get one column of all rounds predictions and labels, in order to calculate
    the per round measures
    :param prediction: pd.Dataframe with the the prediction
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
        flat_data = copy.deepcopy(prediction)
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
        if expanded_df[new_column].dtype == int:
            expanded_df = expanded_df.loc[expanded_df[new_column].isin([0, 1])]
        elif expanded_df[new_column].dtype == str:
            expanded_df = expanded_df.loc[expanded_df[new_column].isin(['0', '1'])]
        else:
            print(f'expanded_df[new_column] type must be int or str')
            raise ValueError(f'expanded_df[new_column] type must be int or str')
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

    return flat_data
