import os
import json
import pprint
import pathlib
import argparse

import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from optlearn.data import data_utils
from optlearn.train import train_utils
from optlearn.train import model_utils


models = {
    "logreg": LogisticRegression,
    "linear_svc": LinearSVC,
    "svc": SVC,
    "knn": KNeighborsClassifier,
    "ridge": RidgeClassifier,
}

param_jsons = {
    "logreg": "jsons/logreg_params.json",
    "linear_svc": "jsons/linear_svc_params.json",
    "svc": "jsons/linear_svc_params.json",
    "knn": "jsons/knn_params.json",
    "ridge": "jsons/ridge_params.json",
}


def select_model(model_name=None):
    """ Select the model, given an identifying string """

    if model_name is None:
        print("No model specified, defaulting to logreg!")
        model_name = "logreg"
        
    try:
        
        model = models[model_name]
    except KeyError:
        print("Model name must be one of {}, defaullting to logreg!".format(models.keys()))
        model = models["logreg"]
    except Exception as exception:
        print("Unknown error when trying to select specified model \"{}\"".format(model_name))
        raise exception
    
    return model


def setup_model(model, model_param_json_path=None):
    """ Set up the model with its params """

    if model_param_json_path is None:
        try:
            model_reverse = {value: key for (key, value) in models.items()}
            model_name = model_reverse.get(model)
            current_directory = pathlib.Path(__file__).parent.absolute()  
            model_param_json_path = os.path.join(current_directory, param_jsons[model_name])
        except KeyError as exception:
            print("Cannot find a default parameter set for the given model!")
            raise exception

    try:
        with open(model_param_json_path, "r") as json_file:
            parameters = json.load(json_file)
    except FileNotFoundError as exception:
        print("Cannot find the file {}!".format(model_param_json_path))
        raise exception
    except Exception as exception:
        print("Unknown error when trying to load parameters from file!")

    try:
        if "class_weight" in parameters:
            parameters["class_weight"] = {key: float(value) for (key, value) in
                                          parameters["class_weight"].items()}
            print(parameters)
            for key in list(parameters["class_weight"].keys()):
                parameters["class_weight"][int(key)] = parameters["class_weight"][key]
                parameters["class_weight"].pop(key)
    except Exception as exception:
        print("Uknown error trying to decode the class weights!")
        print(exception)
        raise exception        
    try:
        model = model(**parameters)
    except Exception as exception:
        print("Unknown error trying to build the model with the given parameters!")
        print(exception)
        raise exception

    return model


def setup_training_params(train_files_parent_path, train_param_json_path=None):
    """ 
        Setup the parameters of the training procedure, with the assumption
        that the features and the labels have been stored as npy files. For
        each feature, there should be a unique directory containing a npy 
        file for each problem instance that has the associated feature for
        each edge of the problem. For any given problem instance, these 
        should have the edge ordering as the same for all features and for
        the labels too.
    """

    if train_param_json_path is None:
        current_directory = pathlib.Path(__file__).parent.absolute()  
        train_param_json_path = os.path.join(current_directory, "jsons/training_params.json")
    try:
        with open(train_param_json_path, "r") as json_file:
            parameters = json.load(json_file)
    except FileNotFoundError as exception:
        print("Cannot find the file {}!".format(train_param_json_path))
        raise exception
    except Exception as exception:
        print("Unknown error when trying to load parameters from file!")
        raise exception

    try:
        parameters["feature_dirs"] = [os.path.join(train_files_parent_path, item)
                                      for item in parameters["feature_dirs"]]
        parameters["solution_dir"] = os.path.join(train_files_parent_path,
                                                   parameters["solution_dir"])
        parameters["weight_dir"] = os.path.join(train_files_parent_path,
                                                   parameters["weight_dir"])
    except Exception as exception:
        print("Unknown error trying to build the features/solution file paths!")
        raise exception
    try:
        if "train_test_val_split" in parameters:
            parameters["train_test_val_split"] = {key: float(value) for (key, value) in
                                          parameters["train_test_val_split"].items()}
    except Exception as exception:
        print("Uknown error trying to decode the test/train/split factors!")
        raise exception

    return parameters


def train_sparsifier(train_files_parent_path, model_name=None,
                     model_param_json_path=None, train_param_json_path=None,
                     model_save_path=None, threshold=None, sample_weights=None):
    """ Train the sparsifier on the MATILDA problem set, given some parameters """

    model = select_model(model_name)
    model = setup_model(model, model_param_json_path)
    train_params = setup_training_params(train_files_parent_path, train_param_json_path)

    if sample_weights is None:
        train_matcher = data_utils.dataMatcher(train_params["feature_dirs"],
                                               train_params["solution_dir"],
        )
        train_tuples = train_matcher.build_fname_pairs()
    else:
        train_matcher = data_utils.dataMatcher(train_params["feature_dirs"],
                                               train_params["solution_dir"],
                                               train_params["weight_dir"],
        )
        train_tuples = train_matcher.build_fname_triples()

    train_loader = data_utils.dataLoader(train_tuples)
    train_loader.train_test_val_split(0.85, 0.10, 0.05)

    if threshold is None:
        print("No threshold given!")
    else:
        threshold = float(threshold)

    if "undersampling_strategy" in train_params:
        sampler = RandomUnderSampler(**train_params["undersampling_strategy"])
    if "oversampling_strategy" in train_params:
        sampler = RandomOverSampler(**train_params["oversampling_strategy"])
    if "oversampling_strategy" in train_params and "undersampling_strategy" in train_params:
        print("Over and undersampling strategies found, defaulting to undersampling!")
        sampler = RandomUnderSampler()
    if "oversampling_strategy" not in train_params and "undersampling_strategy" not in train_params:
        print("Neither over nor undersampling strategies found, defaulting to undersampling!")
        sampler = RandomUnderSampler()

    function_names = ["compute_{}_edges".format(os.path.basename(item))
                      for item in train_params["feature_dirs"]]

    X_train, y_train = train_loader.load_training_features(), train_loader.load_training_labels()
    X_train, y_train = np.vstack(X_train), np.concatenate(y_train)
    # X_train, y_train = sampler.fit_resample(X_train, y_train)

    if sample_weights is not None:
        z_train = np.concatenate(train_loader.load_training_weights())
        print(z_train.shape)
        # z_train, y_train = sampler.fit_resample(z_train, y_train)
        model.fit(X_train, y_train, z_train)
    else:
        model.fit(X_train, y_train)

    wrapper = model_utils.modelWrapper(model=model, function_names=function_names,
                                       threshold=threshold)

    printer = pprint.PrettyPrinter(indent=4)

    X_test, y_test = train_loader.load_testing_features(), train_loader.load_testing_labels()
    X_test, y_test = np.vstack(X_test), np.concatenate(y_test)

    if threshold is not None:
        y_test_pred = (wrapper.predict_proba_vector(X_test) > threshold).astype(int)
    else:
        y_test_pred = wrapper.predict_vector(X_test)
    
    metrics = {"accuracy": train_utils.accuracy(y_test, y_test_pred),
               "fnr": train_utils.false_negative_rate(y_test, y_test_pred),
               "prune": train_utils.pruning_rate(y_test_pred),
    }

    print("Results: ")
    printer.pprint(metrics)

    if model_save_path is not None:            
        persister = model_utils.modelPersister(model=model, function_names=function_names,
                                               threshold=threshold)
        persister.save(model_save_path)
        print("Model saved at {}".format(model_save_path))

    print("Done! :D")
    

if __name__ is not "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", nargs="?", default=None)
    parser.add_argument("-p", "--model_param_json_path", nargs="?", default=None)
    parser.add_argument("-t", "--train_param_json_path", nargs="?", default=None)
    parser.add_argument("-f", "--train_files_parent_path", nargs="?", required=True)
    parser.add_argument("-s", "--model_save_path", nargs="?", default=None)
    parser.add_argument("-y", "--threshold", nargs="?", default=None)
    parser.add_argument("-w", "--sample_weights", nargs="?", default=None)
    
    train_sparsifier(**vars(parser.parse_args()))
