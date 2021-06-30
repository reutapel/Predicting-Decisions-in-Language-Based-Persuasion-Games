import pandas as pd
import ray
import numpy as np
import logging
import os
import json
import utils
from datetime import datetime
import execute_cv_models
import copy
import random
import joblib
import sys
import ast
from os.path import dirname, abspath
import shutil

random.seed(123)

# define directories
base_directory = os.path.abspath(os.curdir)
REVIEWS_FEATURES_DATASETS_DIR = os.path.join(base_directory, 'BERT_HC', 'Reviews_Features', 'datasets')
condition = 'verbal'
data_directory = os.path.join(base_directory, 'data', condition, 'cv_framework')
pair_folds_file_name = 'pairs_folds_new_test_data.csv'

os.environ['http_proxy'] = 'some proxy'
os.environ['https_proxy'] = 'some proxy'

lstm_gridsearch_params = [
    {'lstm_dropout': lstm_dropout, 'linear_dropout': lstm_dropout,
     'lstm_hidden_dim': hidden_size, 'num_layers': num_layers}
    for lstm_dropout in [0.0, 0.1, 0.2, 0.3]
    for hidden_size in [50, 80, 100, 200]
    for num_layers in [1, 2, 3]
]

avg_turn_gridsearch_params = [{'avg_loss': 1.0, 'turn_loss': 1.0, 'avg_turn_loss': 1.0},
                              {'avg_loss': 2.0, 'turn_loss': 2.0, 'avg_turn_loss': 1.0},
                              {'avg_loss': 1.0, 'turn_loss': 1.0, 'avg_turn_loss': 2.0}]


transformer_gridsearch_params = [
    {'num_encoder_layers': num_layers, 'feedforward_hidden_dim_prod': feedforward_hidden_dim_prod,
     'lstm_dropout': transformer_dropout, 'linear_dropout': transformer_dropout}
    for num_layers in [3, 4, 5, 6]
    for transformer_dropout in [0.0, 0.1, 0.2, 0.3]
    for feedforward_hidden_dim_prod in [0.5, 1, 2]
    # for lr in [1e-4]
]

svm_gridsearch_params = [{'kernel': 'poly', 'degree': 3}, {'kernel': 'poly', 'degree': 5},
                         {'kernel': 'poly', 'degree': 8}, {'kernel': 'rbf', 'degree': ''},
                         {'kernel': 'linear', 'degree': ''}]

xgboost_gridsearch_params = [{'learning_rate': learning_rate, 'n_estimators': n_estimators, 'max_depth': max_depth,
                              'min_child_weight': min_child_weight, 'gamma': gamma, 'subsample': subsample}
                             for learning_rate in [0.0, 0.1, 0.2, 0.3]
                             for n_estimators in [50, 80, 100]
                             for max_depth in [2, 3, 4]
                             for min_child_weight in [1, 2]
                             for gamma in [0, 1]
                             for subsample in [1]]

crf_gridsearch_params = [{'squared_sigma': squared_sigma} for squared_sigma in [0.005, 0.006, 0.007, 0.008]]


def execute_create_fit_predict_eval_model(
        function_to_run, model_num, fold, fold_dir, model_type, model_name, data_file_name, fold_split_dict,
        table_writer, hyper_parameters_dict, excel_models_results, all_models_results, model_num_results_path,
        num_iterates=1):
    metadata_dict = {'model_num': model_num, 'model_type': model_type, 'model_name': model_name,
                     'data_file_name': data_file_name, 'hyper_parameters_str': hyper_parameters_dict}
    metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index').T
    model_class = getattr(execute_cv_models, function_to_run)(
        model_num, fold, fold_dir, model_type, model_name, data_file_name, fold_split_dict, table_writer,
        data_directory, hyper_parameters_dict, excel_models_results)
    model_class.load_data_create_model()
    results_df = pd.DataFrame()
    for i in range(num_iterates):
        print(f'Start Iteration number {i}')
        logging.info(f'Start Iteration number {i}')
        model_class.fit_validation()
        results_dict = model_class.eval_model()
        current_results_df = pd.DataFrame.from_dict(results_dict).T
        results_df = pd.concat([results_df, current_results_df], sort='False')
    results_df['raisha_round'] = results_df.index
    results_df[['Raisha', 'Round']] = results_df.raisha_round.str.split(expand=True)
    results_df = results_df.drop('raisha_round', axis=1)
    results_df = results_df.groupby(by=['Raisha', 'Round']).mean()
    results_df = results_df.reset_index()
    results_df.index = np.zeros(shape=(results_df.shape[0],))
    results_df = metadata_df.join(results_df)
    all_models_results = pd.concat([all_models_results, results_df], sort='False')
    model_num_results = joblib.load(model_num_results_path)
    results_for_model_num_results =\
        results_df.loc[(results_df.Raisha == 'All_raishas') & (results_df.Round == 'All_rounds')][
            ['model_num', 'model_name', 'model_type', 'hyper_parameters_str', 'data_file_name', 'RMSE', 'Raisha',
             'Round']].copy(True)
    model_num_results = pd.concat([model_num_results, results_for_model_num_results], sort='False')
    utils.write_to_excel(model_class.model_table_writer, 'Model results', ['Model results'], results_df)
    model_class.model_table_writer.save()
    joblib.dump(model_num_results, model_num_results_path)
    # del model_class

    return all_models_results


@ray.remote
def execute_fold_parallel(participants_fold: pd.Series, fold: int, cuda_device: str,
                          hyper_parameters_tune_mode: bool=False, model_nums_list: list=None,
                          reversed_order: bool=False, bert_hc_exp: bool=False):
    """
    This function get a dict that split the participant to train-val-test (for this fold) and run all the models
    we want to compare --> it train them using the train data and evaluate them using the val data
    :param participants_fold: split the participant to train-val-test (for this fold)
    :param fold: the fold number
    :param cuda_device: the number of cuda device if using it
    :param hyper_parameters_tune_mode: after find good data - hyper parameter tuning
    :param model_nums_list: list of models to run
    :param reversed_order: if to run with reversed_order of the features in the causal graph
    :param bert_hc_exp: if we run the BERt_HC experiment (textual features are created by BERT fine tuning)
    :return:
    """
    # get the train, test, validation participant code for this fold
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    fold_split_dict = dict()
    for data_set in ['train', 'test', 'validation']:
        fold_split_dict[data_set] = participants_fold.loc[participants_fold == data_set].index.tolist()

    # models_to_compare should have for each row:
    # model_num, model_type, model_name, function_to_run, data_file_name, hyper_parameters
    # (strings of all parameters for the running function as dict: {'parameter_name': parameter_value})
    models_to_compare = pd.read_excel(os.path.join(base_directory, 'models_info.xlsx'),
                                      sheet_name='table_to_load', skiprows=[0])
    fold_dir = utils.set_folder(f'fold_{fold}', run_dir)
    excel_models_results = utils.set_folder(folder_name='excel_models_results', father_folder_path=fold_dir)
    # for test
    print(f'test_dir: {test_dir}')
    test_fold_dir = utils.set_folder(f'fold_{fold}', test_dir)
    excel_test_models_results = utils.set_folder(folder_name='excel_best_models_results',
                                                 father_folder_path=test_fold_dir)
    test_participants_fold = pd.read_csv(os.path.join(data_directory, pair_folds_file_name))
    test_participants_fold.index = test_participants_fold.pair_id
    test_table_writer = pd.ExcelWriter(os.path.join(excel_test_models_results, f'Results_test_data_best_models.xlsx'),
                                       engine='xlsxwriter')

    path = f"{REVIEWS_FEATURES_DATASETS_DIR}/experiment_manage.csv"
    experiment_manage_df = pd.read_csv(path)
    bert_models = experiment_manage_df.exp_name.values.tolist()

    table_writer = None
    log_file_name = os.path.join(fold_dir, f'LogFile_fold_{fold}.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file_name,
                        level=logging.DEBUG,
                        format='%(asctime)s: %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )

    if model_nums_list is not None:
        all_model_nums = model_nums_list
    else:
        all_model_nums = list(set(models_to_compare.model_num))

    all_models_results = pd.DataFrame()
    all_models_prediction_results = pd.DataFrame()
    if bert_hc_exp:
        if reversed_order:
            bert_models = reversed(list(enumerate(bert_models)))
        else:
            bert_models = enumerate(bert_models)
    else:
        bert_models = enumerate([''])
    for feature_num, bert_feature in bert_models:
        for model_num in all_model_nums:  # compare all versions of each model type
            num_iterates = 1
            model_type_versions = models_to_compare.loc[models_to_compare.model_num == model_num]
            model_num = f'{model_num}_{feature_num}'
            model_num_results_path = os.path.join(excel_models_results, f'model_num_results_{model_num}.pkl')
            if not os.path.isfile(model_num_results_path):
                model_num_results = pd.DataFrame(columns=['model_num', 'model_name', 'model_type',
                                                          'hyper_parameters_str', 'data_file_name',
                                                          'RMSE', 'Raisha', 'Round'])
                joblib.dump(model_num_results, model_num_results_path)

            for index, row in model_type_versions.iterrows():  # iterate over all the models to compare
                # get all model parameters
                model_type = row['model_type']
                model_name = row['model_name']

                function_to_run = row['function_to_run']
                data_file_name = row['data_file_name']
                test_data_file_name = row['test_data_file_name']

                if bert_hc_exp:
                    model_name = f'{model_name}_{bert_feature}'

                if bert_hc_exp:
                    data_file_name =\
                        data_file_name.replace('bert_embedding', f'bert_embedding_for_feature_{bert_feature}')
                    test_data_file_name =\
                        test_data_file_name.replace('bert_embedding', f'bert_embedding_for_feature_{bert_feature}')
                hyper_parameters_str = row['hyper_parameters']
                # get hyper parameters as dict
                if type(hyper_parameters_str) == str:
                    hyper_parameters_dict = json.loads(hyper_parameters_str)
                else:
                    hyper_parameters_dict = None

                if hyper_parameters_dict is not None and 'features_max_size' in hyper_parameters_dict.keys():
                    if int(hyper_parameters_dict['features_max_size']) > 1000:
                        continue

                if outer_is_debug:
                    hyper_parameters_dict['num_epochs'] = 2
                else:
                    hyper_parameters_dict['num_epochs'] = 100

                # if predict test already done:
                predict_folder = os.path.join(test_dir, f'fold_{fold}',
                                              f'{model_num}_{model_type}_{model_name}_'
                                              f'{hyper_parameters_dict["num_epochs"]}_epochs_fold_num_{fold}')
                if os.path.isdir(predict_folder):
                    continue

                # each function need to get: model_num, fold, fold_dir, model_type, model_name, data_file_name,
                # fold_split_dict, table_writer, data_directory, hyper_parameters_dict.
                # During running it needs to write the predictions to the table_writer and save the trained model with
                # the name: model_name_model_num to the fold_dir.
                # it needs to return a dict with the final results over the evaluation data: {measure_name: measure}
                if hyper_parameters_tune_mode:
                    if 'LSTM' in model_type or 'Transformer' in model_type:
                        if 'LSTM' in model_type and 'use_transformer' not in model_type:
                            greadsearch = lstm_gridsearch_params
                        else:  # for Transformer models and LSTM_use_transformer models
                            greadsearch = transformer_gridsearch_params
                        for i, parameters_dict in enumerate(greadsearch):
                            if outer_is_debug and i > 1:
                                continue
                            new_hyper_parameters_dict = copy.deepcopy(hyper_parameters_dict)
                            new_hyper_parameters_dict.update(parameters_dict)
                            if 'linear' in model_type and 'lstm_hidden_dim' in new_hyper_parameters_dict:
                                new_hyper_parameters_dict['linear_hidden_dim'] = \
                                    int(0.5 * int(new_hyper_parameters_dict['lstm_hidden_dim']))
                            if '_avg_turn' in model_type:
                                for inner_i, inner_parameters_dict in enumerate(avg_turn_gridsearch_params):
                                    if outer_is_debug and inner_i > 1:
                                        continue
                                    new_hyper_parameters_dict.update(inner_parameters_dict)
                                    new_model_name = f'{model_name}'
                                    new_model_num = f'{model_num}_{i}_{inner_i}'
                                    if os.path.isfile(os.path.join(excel_models_results,
                                                                   f'Results_fold_{fold}_model_{new_model_num}.xlsx')):
                                        continue
                                    all_models_results = execute_create_fit_predict_eval_model(
                                        function_to_run, new_model_num, fold, fold_dir, model_type, new_model_name,
                                        data_file_name, fold_split_dict, table_writer, new_hyper_parameters_dict,
                                        excel_models_results, all_models_results, model_num_results_path)
                            else:
                                new_model_name = f'{model_name}'
                                new_model_num = f'{model_num}_{i}'
                                if os.path.isfile(os.path.join(excel_models_results,
                                                               f'Results_fold_{fold}_model_{new_model_num}.xlsx')):
                                    continue
                                all_models_results = execute_create_fit_predict_eval_model(
                                    function_to_run, new_model_num, fold, fold_dir, model_type, new_model_name,
                                    data_file_name, fold_split_dict, table_writer, new_hyper_parameters_dict,
                                    excel_models_results, all_models_results, model_num_results_path)
                    elif 'SVM' in model_type and 'XGBoost' not in model_name or 'Baseline' in model_type:
                        if 'baseline' in model_name or 'Baseline' in model_type:
                            svm_gridsearch_params_inner = [{}]
                        else:
                            svm_gridsearch_params_inner = svm_gridsearch_params
                        if 'EWG' in model_name:
                            num_iterates = 5
                        for i, parameters_dict in enumerate(svm_gridsearch_params_inner):
                            if outer_is_debug and i > 1:
                                continue
                            new_hyper_parameters_dict = copy.deepcopy(hyper_parameters_dict)
                            new_hyper_parameters_dict.update(parameters_dict)
                            new_model_name = f'{model_name}'
                            new_model_num = f'{model_num}_{i}'
                            if os.path.isfile(os.path.join(excel_models_results,
                                                           f'Results_fold_{fold}_model_{new_model_num}.xlsx')):
                                continue
                            all_models_results = execute_create_fit_predict_eval_model(
                                function_to_run, new_model_num, fold, fold_dir, model_type, new_model_name,
                                data_file_name, fold_split_dict, table_writer, new_hyper_parameters_dict,
                                excel_models_results, all_models_results, model_num_results_path,
                                num_iterates=num_iterates)

                    elif 'XGBoost' in model_name:
                        for i, parameters_dict in enumerate(xgboost_gridsearch_params):
                            if outer_is_debug and i > 1:
                                continue
                            new_hyper_parameters_dict = copy.deepcopy(hyper_parameters_dict)
                            new_hyper_parameters_dict.update(parameters_dict)
                            new_model_name = f'{model_name}'
                            new_model_num = f'{model_num}_{i}'
                            if os.path.isfile(os.path.join(excel_models_results,
                                                           f'Results_fold_{fold}_model_{new_model_num}.xlsx')):
                                continue
                            all_models_results = execute_create_fit_predict_eval_model(
                                function_to_run, new_model_num, fold, fold_dir, model_type, new_model_name,
                                data_file_name, fold_split_dict, table_writer, new_hyper_parameters_dict,
                                excel_models_results, all_models_results, model_num_results_path,
                                num_iterates=num_iterates)
                    else:
                        print('Model type must be LSTM-kind, Transformer-kind, or SVM-kind')

                    # select the best hyper-parameters set for this model based on the RMSE
                    model_num_results = joblib.load(model_num_results_path)
                    if model_num_results.empty:
                        continue
                    argmin_index = model_num_results.RMSE.argmin()
                    best_model = model_num_results.iloc[argmin_index]
                    best_model_version_num = best_model.model_num
                    logging.info(f'Best model version for model {model_num}-{model_name} in fold {fold} is: '
                                 f'{best_model_version_num}. Start predict over test data')
                    print(f'Best model version for model {model_num}-{model_name} in fold {fold} is: '
                          f'{best_model_version_num}. Start predict over test data')

                    # predict on test data using the best version of this model
                    test_fold_split_dict = dict()
                    test_pair_ids_in_fold = test_participants_fold[f'fold_{fold}']
                    for data_set in ['train', 'test', 'validation']:
                        test_fold_split_dict[data_set] = \
                            test_pair_ids_in_fold.loc[test_pair_ids_in_fold == data_set].index.tolist()
                    hyper_parameters_str = best_model.hyper_parameters_str
                    model_folder = run_dir
                    if not os.path.exists(os.path.join(base_directory, 'logs', model_folder, f'fold_{fold}')):
                        if not os.path.exists(
                                os.path.join(base_directory, 'logs', f'{model_folder}_best', f'fold_{fold}')):
                            # the folder we need not exists
                            print(f'fold {fold} in folder {model_folder} is not exists')
                            continue
                        else:
                            model_folder = f'{model_folder}_best'
                    # get hyper parameters as dict
                    if type(hyper_parameters_str) == str:
                        hyper_parameters_dict = json.loads(hyper_parameters_str)
                    elif type(hyper_parameters_str) == dict:
                        hyper_parameters_dict = hyper_parameters_str
                    else:
                        hyper_parameters_dict = None
                        print('no hyper parameters dict')

                    num_epochs = hyper_parameters_dict['num_epochs']

                    model_file_name = f'{best_model_version_num}_{model_type}_{model_name}_fold_{fold}.pkl'
                    if function_to_run == 'ExecuteEvalLSTM':
                        inner_model_folder = \
                            f'{best_model_version_num}_{model_type}_{model_name}_{num_epochs}_epochs_fold_num_{fold}'
                    else:
                        inner_model_folder = ''
                    trained_model_dir = os.path.join(base_directory, 'logs', model_folder, f'fold_{fold}',
                                                     inner_model_folder)
                    trained_model = joblib.load(os.path.join(trained_model_dir, model_file_name))

                    metadata_dict = {'model_num': model_num, 'model_type': model_type, 'model_name': model_name,
                                     'data_file_name': data_file_name, 'test_data_file_name': test_data_file_name,
                                     'hyper_parameters_str': hyper_parameters_dict, 'fold': fold,
                                     'best_model_version_num': best_model_version_num}

                    metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index').T
                    model_class = getattr(execute_cv_models, function_to_run)(
                        model_num, fold, test_fold_dir, model_type, model_name, data_file_name, test_fold_split_dict,
                        test_table_writer, data_directory, hyper_parameters_dict, excel_test_models_results,
                        trained_model, trained_model_dir, model_file_name, test_data_file_name, 'test')

                    model_class.load_data_create_model()
                    results_df = pd.DataFrame()
                    for i in range(num_iterates):
                        print(f'Start Test Iteration number {i}')
                        logging.info(f'Start Test Iteration number {i}')
                        model_class.predict()
                        results_dict = model_class.eval_model()
                        current_results_df = pd.DataFrame.from_dict(results_dict).T
                        results_df = pd.concat([results_df, current_results_df], sort='False')

                    results_df['raisha_round'] = results_df.index
                    results_df[['Raisha', 'Round']] = results_df.raisha_round.str.split(expand=True)
                    results_df = results_df.drop('raisha_round', axis=1)
                    results_df = results_df.groupby(by=['Raisha', 'Round']).mean()
                    results_df = results_df.reset_index()
                    results_df.index = np.zeros(shape=(results_df.shape[0],))
                    results_df = metadata_df.join(results_df)
                    all_models_prediction_results = pd.concat([all_models_prediction_results, results_df], sort='False')
                    utils.write_to_excel(model_class.model_table_writer, 'Model results', ['Model results'],
                                         results_df)
                    model_class.model_table_writer.save()

                    model_num_results = model_num_results.reset_index()

                    for remove_index, remove_row in model_num_results.iterrows():
                        if remove_row.model_num == best_model_version_num:
                            continue
                        hyper_parameters_str = remove_row.hyper_parameters_str
                        # get hyper parameters as dict
                        if type(hyper_parameters_str) == str:
                            hyper_parameters_dict = json.loads(hyper_parameters_str)
                        elif type(hyper_parameters_str) == dict:
                            hyper_parameters_dict = hyper_parameters_str
                        else:
                            hyper_parameters_dict = None
                            print('no hyper parameters dict')
                        num_epochs = hyper_parameters_dict['num_epochs']
                        inner_model_folder = f'{remove_row.model_num}_{remove_row.model_type}_' \
                                             f'{remove_row.model_name}_{num_epochs}_epochs_fold_num_{fold}'
                        model_folder = os.path.join(base_directory, 'logs', model_folder, f'fold_{fold}',
                                                    inner_model_folder)
                        if os.path.exists(model_folder):
                            print(f'remove {model_folder}')
                            shutil.rmtree(model_folder)
                        else:
                            print(f'Folder {model_folder} does not exist')

                else:  # no hyper parameters
                    all_models_results = execute_create_fit_predict_eval_model(
                        function_to_run, model_num, fold, fold_dir, model_type, model_name, data_file_name, fold_split_dict,
                        table_writer, hyper_parameters_dict, excel_models_results, all_models_results,
                        model_num_results_path)

    utils.write_to_excel(table_writer, 'All models results', ['All models results'], all_models_results)
    if table_writer is not None:
        table_writer.save()
    if test_table_writer is not None:
        utils.write_to_excel(test_table_writer, 'All models results', ['All models results'],
                             all_models_prediction_results)
        test_table_writer.save()

    logging.info(f'fold {fold} finish compare models')
    print(f'fold {fold} finish compare models')

    return f'fold {fold} finish compare models'


def parallel_main(model_nums_list: list=None, reversed_order: bool=False, cuda: int=None, bert_hc_exp: bool=False):
    print(f'Start run in parallel: for each fold compare all the models')
    logging.info(f'Start run in parallel: for each fold compare all the models')

    # participants_fold_split should have the following columns: fold_0, fold_1,...,fold_5
    # the index should be the participant code
    # the values will be train/test/validation
    participants_fold_split = pd.read_csv(os.path.join(data_directory, pair_folds_file_name))
    participants_fold_split.index = participants_fold_split.pair_id

    if cuda is not None:
        cuda_devices = {0: cuda, 1: cuda,
                        2: cuda, 3: cuda,
                        4: cuda, 5: cuda}
    else:
        cuda_devices = {0: 1, 1: 0,
                        2: 1, 3: 0,
                        4: 1, 5: 0}

    ray.init()

    all_ready_lng =\
        ray.get([execute_fold_parallel.remote(participants_fold_split[f'fold_{i}'], i, str(cuda_devices[i]),
                                              hyper_parameters_tune_mode=True, model_nums_list=model_nums_list,
                                              reversed_order=reversed_order, bert_hc_exp=bert_hc_exp)
                 for i in range(6)])

    print(f'Done! {all_ready_lng}')
    logging.info(f'Done! {all_ready_lng}')

    return


def not_parallel_main(is_debug: bool=False, model_nums_list: list=None, reversed_order: bool=False,
                      bert_hc_exp:bool=False):
    print(f'Start run in parallel: for each fold compare all the models')
    logging.info(f'Start run in parallel: for each fold compare all the models')

    # participants_fold_split should have the following columns: fold_0, fold_1,...,fold_5
    # the index should be the participant code
    # the values will be train/test/validation
    participants_fold_split = pd.read_csv(os.path.join(data_directory, pair_folds_file_name))
    participants_fold_split.index = participants_fold_split.pair_id

    """For debug"""
    if is_debug:
        participants_fold_split = participants_fold_split.iloc[:50]
    for fold in range(1):
        execute_fold_parallel(participants_fold_split[f'fold_{fold}'], fold=fold, cuda_device='1',
                              hyper_parameters_tune_mode=True, model_nums_list=model_nums_list,
                              reversed_order=reversed_order, bert_hc_exp=bert_hc_exp)


if __name__ == '__main__':
    """
    sys.argv[1] = is_parallel: True/False
    sys.argv[2] = folder_date: False or dat in the following format: %d_%m_%Y_%H_%M
    sys.argv[3] = is_debug: True/False
    sys.argv[4] = model_nums_list: False or list of numbers
    sys.argv[5] = reversed order of features: True/False
    sys.argv[6] = outer_cuda: int: 0/1
    sys.argv[7] = bert_hc experiment: True/False
    """

    # is_parallel
    is_parallel = sys.argv[1]
    if is_parallel == 'False':
        is_parallel = False

    run_dir_name = datetime.now().strftime(f'compare_prediction_models_%d_%m_%Y_%H_%M')
    test_dir_name = datetime.now().strftime(f'predict_best_models_%d_%m_%Y_%H_%M')
    if len(sys.argv) > 2:
        folder_date = sys.argv[2]
        if folder_date != 'False':
            run_dir = utils.set_folder(datetime.now().strftime(f'compare_prediction_models_{folder_date}'), 'logs')
            # for test
            test_dir = utils.set_folder(datetime.now().strftime(f'predict_best_models_{folder_date}'), 'logs')
        else:
            # folder dir
            run_dir = utils.set_folder(run_dir_name, 'logs')
            # for test
            test_dir = utils.set_folder(test_dir_name, 'logs')
    else:
        # folder dir
        run_dir = utils.set_folder(run_dir_name, 'logs')
        # for test
        test_dir = utils.set_folder(test_dir_name, 'logs')

    print(f'test_dir: {test_dir}')
    # is_debug
    if len(sys.argv) > 3:
        outer_is_debug = sys.argv[3]
    else:
        outer_is_debug = False
    if outer_is_debug == 'False':
        outer_is_debug = False

    # model_nums_list
    if len(sys.argv) > 4:
        outer_model_nums_list = sys.argv[4]
        if outer_model_nums_list != 'False':
            outer_model_nums_list = ast.literal_eval(outer_model_nums_list)
        else:
            outer_model_nums_list = None
    else:
        outer_model_nums_list = None

    # reversed_order
    if len(sys.argv) > 5:
        outer_reversed_order = sys.argv[5]
    else:
        outer_reversed_order = False
    if outer_reversed_order == 'False':
        outer_reversed_order = False

    # outer_cuda
    if len(sys.argv) > 6:
        outer_cuda = sys.argv[6]
        outer_cuda = int(outer_cuda)
    else:
        outer_cuda = None

    # bert_hc_exp
    if len(sys.argv) > 5:
        bert_hc_exp = sys.argv[5]
    else:
        bert_hc_exp = False
    if bert_hc_exp == 'False':
        bert_hc_exp = False

    # read function
    print(f'run with parameters: is_parallel: {is_parallel}, folder_date: {folder_date}, is_debug: {outer_is_debug},'
          f'model_nums_list: {outer_model_nums_list}, reversed_order: {outer_reversed_order}')
    if is_parallel:
        parallel_main(model_nums_list=outer_model_nums_list, reversed_order=outer_reversed_order, cuda=outer_cuda,
                      bert_hc_exp=bert_hc_exp)
    else:
        # need to put line 96: @ray.remote in comment
        not_parallel_main(outer_is_debug, model_nums_list=outer_model_nums_list, reversed_order=outer_reversed_order,
                          bert_hc_exp=bert_hc_exp)
