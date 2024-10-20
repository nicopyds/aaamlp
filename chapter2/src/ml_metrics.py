import numpy as np


def custom_accuracy(y_true, y_pred):

    assert len(y_true) == len(y_pred), "y_true and y_pred must be equal sizes"

    correct_counter = 0

    for y_true_, y_pred_ in zip(y_true, y_pred):
        if y_true_ == y_pred_:
            correct_counter += 1

    return round(correct_counter / len(y_true), 2)


def custom_tp(y_true, y_pred):

    assert len(y_true) == len(y_pred), "y_true and y_pred must be equal sizes"

    correct_counter = 0

    for y_true_, y_pred_ in zip(y_true, y_pred):

        if y_true_ == 1 and y_pred_ == 1:
            correct_counter += 1

    return correct_counter


def custom_tn(y_true, y_pred):

    assert len(y_true) == len(y_pred), "y_true and y_pred must be equal sizes"

    correct_counter = 0

    for y_true_, y_pred_ in zip(y_true, y_pred):

        if y_true_ == 0 and y_pred_ == 0:
            correct_counter += 1

    return correct_counter


def custom_fp(y_true, y_pred):

    assert len(y_true) == len(y_pred), "y_true and y_pred must be equal sizes"

    correct_counter = 0

    for y_true_, y_pred_ in zip(y_true, y_pred):

        if y_true_ == 0 and y_pred_ == 1:
            correct_counter += 1

    return correct_counter


def custom_fn(y_true, y_pred):

    assert len(y_true) == len(y_pred), "y_true and y_pred must be equal sizes"

    correct_counter = 0

    for y_true_, y_pred_ in zip(y_true, y_pred):

        if y_true_ == 1 and y_pred_ == 0:
            correct_counter += 1

    return correct_counter


def custom_accuracy_2(y_true, y_pred):

    assert len(y_true) == len(y_pred), "y_true and y_pred must be equal sizes"

    tp = custom_tp(y_true, y_pred)
    tn = custom_tn(y_true, y_pred)
    fp = custom_fp(y_true, y_pred)
    fn = custom_fn(y_true, y_pred)

    return round(((tp + tn) / (tp + tn + fp + fn)), 2)


def custom_precision(y_true, y_pred):

    assert len(y_true) == len(y_pred), "y_true and y_pred must be equal sizes"

    tp = custom_tp(y_true, y_pred)
    fp = custom_fp(y_true, y_pred)

    return round(((tp) / (tp + fp)), 2)


def custom_recall(y_true, y_pred):

    assert len(y_true) == len(y_pred), "y_true and y_pred must be equal sizes"

    tp = custom_tp(y_true, y_pred)
    fn = custom_fn(y_true, y_pred)

    return round(((tp) / (tp + fn)), 2)


def custom_pr_from_proba(y_true, y_pred_proba):

    assert len(y_true) == len(y_pred_proba), "y_true and y_pred must be equal sizes"

    precision = []
    recall = []

    thresholds = np.arange(0.1, 1.1, 0.1)

    for threshold in thresholds:

        y_pred = [
            1 if y_pred_proba_ >= threshold else 0 for y_pred_proba_ in y_pred_proba
        ]

        p = custom_precision(y_true=y_true, y_pred=y_pred)
        r = custom_recall(y_true=y_true, y_pred=y_pred)

        precision.append(p)
        recall.append(r)

    return precision, recall
