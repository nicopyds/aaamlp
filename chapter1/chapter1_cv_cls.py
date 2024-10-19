#!/usr/bin/env python3

from argparse import ArgumentParser

import os
import numpy as np
import pandas as pd

from sklearn import set_config

set_config(transform_output="pandas")

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    train_test_split,
    StratifiedGroupKFold,
)

MAPPINGS = {
    "kfold": KFold,
    "skfold": StratifiedKFold,
    "l1o": LeaveOneOut,
    "sgkfold": StratifiedGroupKFold,
    "hold": train_test_split,
}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_dummy_data():

    X = pd.DataFrame(np.ones((17, 2)), columns=["X", "X1"])
    y = pd.DataFrame(
        np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), columns=["y"]
    )
    groups = pd.DataFrame(
        np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8]),
        columns=["groups"],
    )

    X = pd.concat([X, y, groups], axis=1)

    return X


def load_wine():

    X = pd.read_csv(os.path.join(path, "red.csv"), sep=";")
    X["quality"] = X["quality"] - 3

    return X


def assert_cv_strategies(cv):
    assert_message = (
        "You have to supply one of the 5 possible cv strategies:{'.'.join(MAPPINGS.keys())}",
    )
    assert cv in MAPPINGS.keys(), assert_message


def report_cv(X, sgkfold=False):

    if sgkfold:
        pt = X.pivot_table(
            index=["groups", "fold"], columns="y", values="X", aggfunc=len, margins=True
        )
        print(pt)
        return

    r = X.groupby(["fold"])["quality"].value_counts().unstack()

    # estamos en un caso de leave one out cv
    if len(r) == X.shape[0]:
        r = r.sum(axis=1)
        print(r)
        print(sum(r > 1))

    else:
        print(r)


def add_kfolds():

    X = load_wine()
    X["fold"] = -1

    kfold = KFold(n_splits=5, shuffle=True, random_state=175)

    for split, (_, test_idx) in enumerate(kfold.split(X), start=1):
        X.loc[test_idx, "fold"] = split

    report_cv(X=X)

    return X


def add_hold_out_fold():

    X = load_wine()
    X["fold"] = -1

    X_train, X_test = train_test_split(
        X, test_size=0.25, shuffle=True, random_state=175
    )

    X.loc[X_train.index, "fold"] = 1
    X.loc[X_test.index, "fold"] = 2

    report_cv(X=X)

    return X


def add_leave_one_out_fold():

    X = load_wine()
    X["fold"] = -1
    loofold = LeaveOneOut()

    # devuelve un numpy array
    for i, (_, X_test) in enumerate(loofold.split(X), start=1):

        X.loc[X_test, "fold"] = i

    report_cv(X=X)

    return X


def add_stratified_kfolds():

    X = load_wine()
    y = X["quality"]

    X["fold"] = -1

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=175)

    for split, (_, test_idx) in enumerate(skfold.split(X, y), start=1):
        X.loc[test_idx, "fold"] = split

    report_cv(X=X)

    return X


def add_stratified_group_kfolds():

    X = get_dummy_data()
    y = X["y"]
    groups = X["groups"]

    X["fold"] = -1

    sgkfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=175)

    for split, (_, test_idx) in enumerate(sgkfold.split(X, y, groups), start=1):
        X.loc[test_idx, "fold"] = split

    report_cv(X=X, sgkfold=True)

    return X


def main(cv: str):
    assert_cv_strategies(cv=cv)

    if cv == "kfold":
        X = add_kfolds()

    if cv == "skfold":
        X = add_stratified_kfolds()

    if cv == "hold":
        X = add_hold_out_fold()

    if cv == "l1o":
        X = add_leave_one_out_fold()

    if cv == "sgkfold":
        X = add_stratified_group_kfolds()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-cv",
        "--cross_validation_type",
        help=f"You have to supply one of the 5 possible cv strategies:{'.'.join(MAPPINGS.keys())}",
        required=True,
    )

    cv = vars(parser.parse_args())["cross_validation_type"]

    main(cv=cv)
