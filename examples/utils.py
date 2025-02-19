#!/usr/bin/env python
import numpy as np
from sklearn.metrics import confusion_matrix


def print_data_stats(sens_attr, class_labels):
    """Print a few numbers about the data: Total number of points, number of
    protected examples and unprotected examples, and number of protected points
    in positive class, and number of unprotected points in positive class.

    Parameters
    -----------
    sens_attr: numpy array
     The sensitive attribute of shape=(number_points,).
    class_labels: nunmp
        The class labels of shape=(number_points,).
    """
    non_prot_all = sum(sens_attr == 1.0)  # non-protected group
    prot_all = len(sens_attr) - non_prot_all  # protected group
    non_prot_pos = sum(
        class_labels[sens_attr == 1.0] == 1.0
    )  # non_protected in positive class
    prot_pos = sum(class_labels == 1.0) - non_prot_pos  # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)
    print
    print("Total data points: %d" % len(sens_attr))
    print("# non-protected examples: %d" % non_prot_all)
    print("# protected examples: %d" % prot_all)
    print(
        "# non-protected examples in positive class: %d (%0.1f%%)"
        % (non_prot_pos, non_prot_pos * 100.0 / non_prot_all)
    )
    print(
        "# protected examples in positive class: %d (%0.1f%%)"
        % (prot_pos, prot_pos * 100.0 / prot_all)
    )


def get_positive_rate(y_predicted, y_true):
    """Compute the positive rate for given predictions of the class label.

    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).

    Returns
    ---------
    pr: float
        The positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    pr = (tp + fp) / (tp + fp + tn + fn)
    return pr


def get_true_positive_rate(y_predicted, y_true):
    """Compute the true positive rate for given predictions of the class label.

    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).

    Returns
    ---------
    tpr: float
        The true positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    tpr = tp / (tp + fn)
    return tpr


def compute_fairness_measures(y_predicted, y_true, sens_attr):
    """Compute value of demographic parity and equality of opportunity for given predictions.

    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).
    sens_attr: numpy array
        The sensitive labels of shape=(number_points,).

    Returns
    ----------
    DDP: float
        The difference of demographic parity.
    DEO: float
        The difference of equality of opportunity.
    """
    positive_rate_prot = get_positive_rate(
        y_predicted[sens_attr == -1], y_true[sens_attr == -1]
    )
    positive_rate_unprot = get_positive_rate(
        y_predicted[sens_attr == 1], y_true[sens_attr == 1]
    )
    true_positive_rate_prot = get_true_positive_rate(
        y_predicted[sens_attr == -1], y_true[sens_attr == -1]
    )
    true_positive_rate_unprot = get_true_positive_rate(
        y_predicted[sens_attr == 1], y_true[sens_attr == 1]
    )
    DDP = positive_rate_unprot - positive_rate_prot
    DEO = true_positive_rate_unprot - true_positive_rate_prot

    return DDP, DEO


def get_ddp_linear(
    y_reg,
    sens_attr,
):
    """Compute the demographic parity for given predictions.

    Parameters
    ----------
    y_reg: numpy array
        The value of f on x of shape=(number_points,).
    sens_attr: numpy array
        The sensitive labels of shape=(number_points,).

    Returns
    ---------
    DDP: float
        The linear relaxation of demographic parity as defined by Donini et al. (2018).
    """
    n = len(y_reg)
    p1_hat = (sens_attr == 1).sum() / n
    return (
        1
        / n
        * np.sum(
            [
                sens_attr[i]
                * y_reg[i]
                / (p1_hat if sens_attr[i] == 1.0 else 1.0 - p1_hat)
                for i in range(n)
            ]
        )
    )


def get_ddp_convex_concave(y_reg, sens_attr):
    """Compute the demographic parity for given predictions.

    Parameters
    ----------
    y_reg: numpy array
        The value of f on x of shape=(number_points,).
    sens_attr: numpy array
        The sensitive labels of shape=(number_points,).

    Returns
    ---------
    DDP: float
        The convex concave relaxation of demographic parity as defined by Zafar et al. (2017a).
    """
    n = len(y_reg)
    p1_hat = (sens_attr == 1).sum() / n
    return (
        1
        / n
        * np.sum(
            [
                ((sens_attr[i] + 1 / 2) - p1_hat)
                * max(0, y_reg[i])
                / (p1_hat * (1.0 - p1_hat))
                for i in range(n)
            ]
        )
    )


def get_ddp_upper(y_reg, sens_attr):
    """Compute the demographic parity for given predictions.

    Parameters
    ----------
    y_reg: numpy array
        The value of f on x of shape=(number_points,).
    sens_attr: numpy array
        The sensitive labels of shape=(number_points,).

    Returns
    ---------
    DDP: float
        The upper relaxation of demographic parity as defined by Wu et al. (2019).
    """
    n = len(y_reg)
    p1_hat = (sens_attr == 1).sum() / n
    kappa = lambda x: max(0, 1 + x)
    return (
        1
        / n
        * np.sum(
            [
                (
                    kappa(y_reg[i]) / p1_hat
                    if sens_attr[i] == 1.0
                    else kappa(-y_reg[i]) / (1.0 - p1_hat)
                )
                - 1
                for i in range(n)
            ]
        )
    )


def get_ddp_lower(y_reg, sens_attr):
    """Compute the demographic parity for given predictions.

    Parameters
    ----------
    y_reg: numpy array
        The value of f on x of shape=(number_points,).
    sens_attr: numpy array
        The sensitive labels of shape=(number_points,).

    Returns
    ---------
    DDP: float
        The lower relaxation of demographic parity as defined by Wu et al. (2019).
    """
    n = len(y_reg)
    p1_hat = (sens_attr == 1).sum() / n
    delta = lambda x: min(1, x)
    return (
        1
        / n
        * np.sum(
            [
                (
                    delta(y_reg[i]) / p1_hat
                    if sens_attr[i] == 1.0
                    else delta(-y_reg[i]) / (1.0 - p1_hat)
                )
                - 1
                for i in range(n)
            ]
        )
    )


def normalize_by_scaling(matrix: np.array):
    """Normalize the matrix by scaling the values to be between -1 and 1.

    Parameters
    ----------
    matrix: numpy array
        The matrix to be normalized.

    Returns
    ----------
    matrix: numpy array
        The normalized matrix.
    """
    return matrix / np.max(np.abs(matrix))


def get_accuracy(y_true, y_predicted):
    """Compute the accuracy for given predicted class labels.

    Parameters
    ----------
    y_true: numpy array
        The true class labels of shape=(number_points,).
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).

    Returns
    ---------
    accuracy: float
        The accuracy of the predictions.
    """
    correct_answers = (y_predicted == y_true).astype(
        int
    )  # will have 1 when the prediction and the actual label match
    accuracy = float(sum(correct_answers)) / float(len(correct_answers))
    return accuracy
