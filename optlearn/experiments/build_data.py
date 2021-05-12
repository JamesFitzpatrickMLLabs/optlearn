import os
import argparse

from optlearn.data import data_utils


def build_features(numpy_dir, problem_dir, features):
    """ 
    Given a directory to write numpy files to and a directory from which to pick up
    problem instances, build the specified features and write them into separate npy
    files for feature for eahc problem instance.
    """

    problems = os.listdir(problem_dir)
    problems = [os.path.join(problem_dir, problem) for problem in problems]
    solutions = [None] * len(problems)
    file_pairs = list(zip(problems, solutions))

    features = ["compute_{}_edges".format(item) for item in features]


    builder = data_utils.createTrainingFeatures(numpy_dir, features,
                                                file_pairs, verbose=True)
    builder.data_create()

    print("Done! :D")


if __name__ is not "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numpy_dir", nargs="?", required=True)
    parser.add_argument("-p", "--problem_dir", nargs="?", required=True)
    parser.add_argument("-f", "--features", nargs="+", required=True)
    
    build_features(**vars(parser.parse_args()))
