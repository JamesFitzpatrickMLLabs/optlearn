import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def accuracy(y, y_hat):
    """ Compute the accuracy """

    return accuracy_score(y, y_hat)


def true_positive_rate(y, y_hat):
    """ Compute the true positive rate """

    return recall_score(y, y_hat)


def false_negative_rate(y, y_hat):
    """ Compute the false negative rate """

    return 1 - true_positive_rate(y, y_hat)


def pruning_rate(y_hat):
    """ Compute the pruning rate for binayr predictions """

    return np.sum(y_hat) / len(y_hat)
