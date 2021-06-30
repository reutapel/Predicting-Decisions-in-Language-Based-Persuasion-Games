import pandas as pd
import ray
import logging
import os
import json
import utils
from datetime import datetime
import predictive_models
from collections import defaultdict
import random
import joblib
import sys
import XAI_Methods
from pathlib import Path
from typing import Union

random.seed(123)

# define directories
base_directory = os.path.abspath(os.curdir)
condition = 'verbal'
data_directory = os.path.join(base_directory, 'data', condition, 'models_input')
pair_folds_file_name = 'reviews_folds.csv'

measures = {'regression': [['RMSE', 'MSE', 'MAE'], 'calculate_continues_predictive_model_measures'],
            'classification': [['Accuracy', 'F-score'], 'calculate_predictive_model_measures']}

os.environ['http_proxy'] = 'some proxy'
os.environ['https_proxy'] = 'some proxy'

default_gridsearch_params = {
    'RandomForest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
    'XGBoost': {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3, 'min_child_weight': 1,
                'gamma': 0, 'subsample': 1},
    'lightGBM': {'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample_for_bin': 50,
                 'min_child_samples': 20, 'reg_alpha': 0., 'reg_lambda': 0.},
    'CatBoost': {'iterations': 500, 'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 3.0},
    'SVM': {'kernel': 'rbf', 'degree': 3},
    'most_frequent': {},
    'mean': {},
    'median': {}

}

gridsearch_params = {
    'RandomForest': [{'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
                     for n_estimators in [50, 80, 100]
                     for max_depth in [None, 3, 5, 8]
                     for min_samples_split in [2, 3]
                     for min_samples_leaf in [1, 2]],
    'XGBoost': [{'learning_rate': learning_rate, 'n_estimators': n_estimators, 'max_depth': max_depth,
                 'min_child_weight': min_child_weight, 'gamma': gamma, 'subsample': subsample}
                for learning_rate in [0.0, 0.1, 0.2, 0.3]
                for n_estimators in [50, 80, 100]
                for max_depth in [2, 3, 4]
                for min_child_weight in [1, 2]
                for gamma in [0, 1]
                for subsample in [1]],
    'lightGBM': [{'num_leaves': num_leaves, 'max_depth': max_depth, 'learning_rate': learning_rate,
                 'n_estimators': n_estimators, 'subsample_for_bin': subsample_for_bin,
                  'min_child_samples': min_child_samples, 'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda}
                 for num_leaves in [0.0, 0.1, 0.2, 0.3]
                 for max_depth in [50, 80, 100, 200]
                 for learning_rate in [1, 2, 3]
                 for n_estimators in [1, 2, 3]
                 for subsample_for_bin in [1, 2, 3]
                 for min_child_samples in [1, 2, 3]
                 for reg_alpha in [1, 2, 3]
                 for reg_lambda in [1, 2, 3]],
    'CatBoost': [{'iterations': iterations, 'depth': depth, 'learning_rate': learning_rate,
                 'l2_leaf_reg': l2_leaf_reg}
                 for iterations in [100, 500]
                 for depth in [6, 10]
                 for learning_rate in [0.03, 0.05]
                 for l2_leaf_reg in [3.0, 1.0]],
    'SVM': [{'kernel': 'poly', 'degree': 3}, {'kernel': 'poly', 'degree': 5}, {'kernel': 'poly', 'degree': 8},
            {'kernel': 'rbf', 'degree': 3}, {'kernel': 'linear', 'degree': 3}],
    'most_frequent': [{}],
    'mean': [{}],
    'median': [{}]
}


def execute_create_fit_predict_eval_model(model_num, features, train_x, train_y, test_x, test_y,
                                          fold, fold_dir, model_name, excel_models_results_folder, model_type,
                                          hyper_parameters_dict, all_models_results, model_num_results_path):
    metadata_dict = {'model_num': model_num, 'model_name': model_name,
                     'hyper_parameters_str': hyper_parameters_dict}
    metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index').T
    print('Create model')
    model_class = predictive_models.PredictiveModel(features, model_name, hyper_parameters_dict, model_num, fold,
                                                    fold_dir, excel_models_results_folder, model_type=model_type)
    print('Fit model')
    model_class.fit(train_x, train_y)
    print('Predict model')
    predictions = model_class.predict(test_x, test_y)
    results_dict = getattr(utils, measures[model_type][1])(all_predictions=predictions)
    results_df = pd.DataFrame(results_dict, index=[0])
    results_df = metadata_df.join(results_df)
    all_models_results = pd.concat([all_models_results, results_df], sort='False')

    utils.write_to_excel(model_class.model_table_writer, 'Model results', ['Model results'], results_df)
    model_class.model_table_writer.save()

    model_num_results = joblib.load(model_num_results_path)
    results_for_model_num_results = results_df.copy(True)
    model_num_results = pd.concat([model_num_results, results_for_model_num_results], sort='False')
    joblib.dump(model_num_results, model_num_results_path)
    del model_class

    return all_models_results


@ray.remote
def execute_fold_parallel(participants_fold: pd.Series, fold: int, cuda_device: str, data_file_name: str,
                          features_families: list, hyper_parameters_tune_mode: bool=False,
                          test_data_file_name: str=None, id_column: str='pair_id', model_type: str='regression',
                          features_to_remove: Union[list, str] = None):
    """
    This function get a dict that split the participant to train-val-test (for this fold) and run all the models
    we want to compare --> it train them using the train data and evaluate them using the val data
    :param participants_fold: split the participant to train-val-test (for this fold)
    :param fold: the fold number
    :param cuda_device: the number of cuda device if using it
    :param hyper_parameters_tune_mode: after find good data - hyper parameter tuning
    :param data_file_name: the data file name
    :param features_families: the families of features to use
    :param id_column: the name of the ID column
    :param test_data_file_name: the test_data_file_name
    :param model_type: is this a regression model or a classification model
    :param features_to_remove: features we want to remove
    :return:
    """
    # get the train, test, validation participant code for this fold
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    fold_split_dict = dict()
    for data_set in ['train', 'test', 'validation']:
        fold_split_dict[data_set] = participants_fold.loc[participants_fold == data_set].index.tolist()

    fold_dir = utils.set_folder(f'fold_{fold}', run_dir)
    excel_models_results = utils.set_folder(folder_name='excel_models_results', father_folder_path=fold_dir)
    # for test
    test_fold_dir = utils.set_folder(f'fold_{fold}', test_dir)
    excel_test_models_results = utils.set_folder(folder_name='excel_best_models_results',
                                                 father_folder_path=test_fold_dir)
    test_participants_fold = pd.read_csv(os.path.join(data_directory, pair_folds_file_name))
    test_participants_fold.index = test_participants_fold[id_column]
    test_table_writer = pd.ExcelWriter(os.path.join(excel_test_models_results, f'Results_test_data_best_models.xlsx'),
                                       engine='xlsxwriter')

    log_file_name = os.path.join(fold_dir, f'LogFile_fold_{fold}.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file_name,
                        level=logging.DEBUG,
                        format='%(asctime)s: %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )

    all_models_results = pd.DataFrame()
    all_results_table_writer = pd.ExcelWriter(os.path.join(excel_models_results,
                                                           f'Results_fold_{fold}_all_models.xlsx'), engine='xlsxwriter')
    all_models_test_data_results = pd.DataFrame()
    best_models_paths_dict = defaultdict(str)

    # load data
    data_path = os.path.join(base_directory, 'data', 'verbal', 'models_input', data_file_name)
    if test_data_file_name is None:
        test_data_path = data_path
    else:
        test_data_path = os.path.join(base_directory, 'data', 'verbal', 'models_input', test_data_file_name)
    train_pair_ids = participants_fold.loc[participants_fold == 'train'].index.tolist()
    validation_pair_ids = participants_fold.loc[participants_fold == 'validation'].index.tolist()
    test_pair_ids = participants_fold.loc[participants_fold == 'test'].index.tolist()

    train_x, train_y, validation_x, validation_y = utils.load_data(data_path=data_path, label_name='label',
                                                                   features_families=features_families,
                                                                   test_pair_ids=validation_pair_ids,
                                                                   train_pair_ids=train_pair_ids, id_column=id_column,
                                                                   features_to_remove=features_to_remove)
    _, _, test_x, test_y = utils.load_data(data_path=test_data_path, label_name='label', id_column=id_column,
                                           features_families=features_families, test_pair_ids=test_pair_ids,
                                           features_to_remove=features_to_remove)

    data_features = train_x.columns.tolist()

    model_names = ['SVM', 'mean', 'median', 'RandomForest', 'XGBoost', 'CatBoost']  # , 'lightGBM', '']

    for model_num, model_name in enumerate(model_names):
        model_num_results_path = os.path.join(excel_models_results, f'model_name_results_{model_name}.pkl')
        if not os.path.isfile(model_num_results_path):
            model_num_results = pd.DataFrame(columns=['model_name', 'hyper_parameters_str'] + measures[model_type][0])
            joblib.dump(model_num_results, model_num_results_path)

        # each function need to get: model_num, fold, fold_dir, model_type, model_name,
        # fold_split_dict, table_writer, data_directory, hyper_parameters_dict.
        # During running it needs to write the predictions to the table_writer and save the trained model with
        # the name: model_name_model_num to the fold_dir.
        # it needs to return a dict with the final results over the evaluation data: {measure_name: measure}
        if hyper_parameters_tune_mode:
            greadsearch = gridsearch_params[model_name]
            for i, parameters_dict in enumerate(greadsearch):
                # if i > 0:
                #     continue
                if os.path.isfile(os.path.join(excel_models_results, f'Results_fold_{fold}_model_{model_name}.xlsx')):
                    continue
                new_model_num = f'{model_num}_{i}'
                print(f'start model {model_name} with number {new_model_num} for fold {fold}')
                all_models_results = execute_create_fit_predict_eval_model(
                    model_num=new_model_num, features=data_features, train_x=train_x, train_y=train_y,
                    test_x=validation_x, test_y=validation_y, fold=fold, fold_dir=fold_dir, model_name=model_name,
                    excel_models_results_folder=excel_models_results, hyper_parameters_dict=parameters_dict,
                    all_models_results=all_models_results, model_num_results_path=model_num_results_path,
                    model_type=model_type)

        else:  # no hyper parameters
            parameters_dict = default_gridsearch_params[model_name]
            all_models_results = execute_create_fit_predict_eval_model(
                model_num=model_num, features=data_features, train_x=train_x, train_y=train_y,
                test_x=validation_x, test_y=validation_y, fold=fold, fold_dir=fold_dir, model_name=model_name,
                excel_models_results_folder=excel_models_results, hyper_parameters_dict=parameters_dict,
                all_models_results=all_models_results, model_num_results_path=model_num_results_path,
                model_type=model_type)

        # select the best hyper-parameters set for this model based on the Accuracy
        model_num_results = joblib.load(model_num_results_path)
        if model_num_results.empty:
            continue
        # measures[model_type][0] is the measure to choose the best model
        if model_type == 'regression':
            argmax_index = model_num_results[measures[model_type][0][0]].argmin()
        elif model_type == 'classification':
            argmax_index = model_num_results[measures[model_type][0][0]].argmax()
        else:
            raise ValueError('model_type must be regression or classification')
        best_model = model_num_results.iloc[argmax_index]
        model_version_num = best_model.model_num
        logging.info(f'Best model version for model {model_num}-{model_name} in fold {fold} is: '
                     f'{model_version_num}. Start predict over test data')
        print(f'Best model version for model {model_num}-{model_name} in fold {fold} is: '
              f'{model_version_num}. Start predict over test data')
        # predict on test data using the best version of this model
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

        model_file_name = f'{model_version_num}_{model_name}_fold_{fold}.pkl'
        trained_model_dir = os.path.join(base_directory, 'logs', model_folder, f'fold_{fold}')
        trained_model = joblib.load(os.path.join(trained_model_dir, model_file_name))
        best_models_paths_dict[model_name] = os.path.join(trained_model_dir, model_file_name)

        metadata_dict = {'model_num': model_num, 'model_name': model_name,
                         'hyper_parameters_str': hyper_parameters_dict, 'fold': fold,
                         'best_model_version_num': model_version_num}

        metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index').T

        # create model class with trained_model
        test_model_class = predictive_models.PredictiveModel(
            data_features, model_name, hyper_parameters_dict, model_num, fold, fold_dir,
            excel_test_models_results, trained_model=trained_model, model_type=model_type)

        test_predictions = test_model_class.predict(test_x, test_y)
        results_dict = getattr(utils, measures[model_type][1])(all_predictions=test_predictions)
        results_df = pd.DataFrame(results_dict, index=[0])
        results_df = metadata_df.join(results_df)
        all_models_test_data_results = pd.concat([all_models_test_data_results, results_df], sort='False')
        utils.write_to_excel(test_model_class.model_table_writer, 'Model results', ['Model results'],
                             results_df)
        test_model_class.model_table_writer.save()

    utils.write_to_excel(all_results_table_writer, 'All models results', ['All models results'], all_models_results)
    if all_results_table_writer is not None:
        all_results_table_writer.save()
    if test_table_writer is not None:
        utils.write_to_excel(test_table_writer, 'All models results', ['All models results'],
                             all_models_test_data_results)
        test_table_writer.save()

    logging.info(f'fold {fold} finish compare models')
    print(f'fold {fold} finish compare models')

    for model_type in best_models_paths_dict.keys():
        if model_type not in ['RandomForest', 'XGBoost', 'CatBoost']:
            continue
        print(f'\n computing SHAP values of {model_type}')
        pkl_model_path = Path(best_models_paths_dict[model_type])
        model = joblib.load(pkl_model_path)
        X_test = train_x
        X_train = train_x

        # create a file for the SHAP results to be saved at
        save_shap_values_path = pkl_model_path.parent.joinpath('SHAP_values_results')
        save_shap_values_path.mkdir(exist_ok=True)

        shap_obj = XAI_Methods.XAIMethods(model, X_test, X_train, 'SHAP', model_type)
        shap_res = shap_obj.get_shap_feature_mean_values()
        shap_res_save_path = save_shap_values_path.joinpath(pkl_model_path.name.replace('pkl', 'csv'))
        shap_res.to_csv(shap_res_save_path)

    return f'fold {fold} finish compare models', best_models_paths_dict


def parallel_main(data_file_name: str, features_families: list, test_data_file_name: str, model_type: str,
                  features_to_remove: Union[list, str] = None):
    print(f'Start run in parallel: for each fold compare all the models')
    logging.info(f'Start run in parallel: for each fold compare all the models')

    # participants_fold_split should have the following columns: fold_0, fold_1,...,fold_5
    # the index should be the participant code
    # the values will be train/test/validation
    participants_fold_split = pd.read_csv(os.path.join(data_directory, pair_folds_file_name))
    if 'review_id' in participants_fold_split.columns:
        id_column = 'review_id'
        participants_fold_split.index = participants_fold_split.review_id
    else:
        participants_fold_split.index = participants_fold_split.pair_id
        id_column = 'pair_id'

    cuda_devices = {0: 0, 1: 1,
                    2: 0, 3: 1,
                    4: 0, 5: 1}

    ray.init()

    all_ready_lng, models_paths_dict =\
        ray.get([execute_fold_parallel.remote(participants_fold_split[f'fold_{i}'], i, str(cuda_devices[i]),
                                              hyper_parameters_tune_mode=True, data_file_name=data_file_name,
                                              test_data_file_name=test_data_file_name, model_type=model_type,
                                              features_families=features_families, id_column=id_column,
                                              features_to_remove=features_to_remove)
                 for i in range(6)])

    print(f'Done! {all_ready_lng}')
    logging.info(f'Done! {all_ready_lng}')

    return models_paths_dict


def not_parallel_main(data_file_name: str, test_data_file_name: str, features_families: list, model_type: str,
                      is_debug: bool=False, num_folds: int=1, features_to_remove: Union[list, str] = None):
    print(f'Start run in parallel: for each fold compare all the models')
    logging.info(f'Start run in parallel: for each fold compare all the models')

    # participants_fold_split should have the following columns: fold_0, fold_1,...,fold_5
    # the index should be the participant code
    # the values will be train/test/validation
    participants_fold_split = pd.read_csv(os.path.join(data_directory, pair_folds_file_name))
    if 'review_id' in participants_fold_split.columns:
        id_column = 'review_id'
        participants_fold_split.index = participants_fold_split.review_id
    else:
        participants_fold_split.index = participants_fold_split.pair_id
        id_column = 'pair_id'
    models_paths_dict = None

    """For debug"""
    if is_debug:
        participants_fold_split = participants_fold_split.iloc[:50]

    for fold in range(num_folds):
        _, models_paths_dict =\
            execute_fold_parallel(participants_fold_split[f'fold_{fold}'], fold=fold, cuda_device='1',
                                  hyper_parameters_tune_mode=False, data_file_name=data_file_name,
                                  test_data_file_name=test_data_file_name, features_families=features_families,
                                  id_column=id_column, model_type=model_type, features_to_remove=features_to_remove)

    if models_paths_dict is None:
        print('No model paths were saved')

    return models_paths_dict


if __name__ == '__main__':
    """
    sys.argv[1] = is_parallel
    sys.argv[2] = outer_data_file_name
    sys.argv[3] = outer_test_data_file_name
    sys.argv[4] = outer_features_families
    sys.argv[5] = folder_date
    sys.argv[6] = is_debug
    sys.argv[7] = model_type: regression/ classification
    """

    # is_parallel
    if len(sys.argv) > 1:
        is_parallel = sys.argv[1]
    else:
        is_parallel = False
    if is_parallel == 'False':
        is_parallel = False

    # outer_data_file_name
    if len(sys.argv) > 2:
        outer_data_file_name = sys.argv[2]
    else:
        outer_data_file_name =\
            "all_data_proportion_label_['hand_crafted_features']_use_decision_features_verbal_test_data.pkl"

    # outer_test_data_file_name
    if len(sys.argv) > 3:
        outer_test_data_file_name = sys.argv[3]
    else:
        # test data
        outer_test_data_file_name =\
            "all_data_proportion_label_['hand_crafted_features']_use_decision_features_verbal_test_data.pkl"

    # outer_features_families
    if len(sys.argv) > 4:
        outer_features_families = sys.argv[4]
    else:
        # the options are: 'history_behave_features', 'history_text_features', 'current_text_features'
        outer_features_families = ['text_features']

    run_dir_name =\
        datetime.now().strftime(f'compare_prediction_models_features_{outer_features_families}_%d_%m_%Y_%H_%M')
    test_dir_name = datetime.now().strftime(f'predict_best_models_features_{outer_features_families}_%d_%m_%Y_%H_%M')
    if len(sys.argv) > 5:
        folder_date = sys.argv[5]
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

    # is_debug
    if len(sys.argv) > 6:
        outer_is_debug = sys.argv[6]
    else:
        outer_is_debug = False
    if outer_is_debug == 'False':
        outer_is_debug = False

    # outer_model_type
    if len(sys.argv) > 7:
        outer_model_type = sys.argv[7]
    else:
        outer_model_type = 'regression'

    features_to_remove = None

    # read function
    if is_parallel:
        best_models_paths_dict =\
            parallel_main(features_families=outer_features_families, data_file_name=outer_data_file_name,
                          test_data_file_name=outer_test_data_file_name, model_type=outer_model_type,
                          features_to_remove=features_to_remove)
    else:
        best_models_paths_dict =\
            not_parallel_main(is_debug=outer_is_debug, features_families=outer_features_families,
                              data_file_name=outer_data_file_name, test_data_file_name=outer_test_data_file_name,
                              model_type=outer_model_type, features_to_remove=features_to_remove)

