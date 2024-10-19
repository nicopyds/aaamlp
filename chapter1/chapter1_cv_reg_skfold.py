#!/usr/bin/env python3

from argparse import ArgumentParser

import os
import numpy as np
import pandas as pd

from sklearn import set_config

set_config(transform_output="pandas")

from sklearn.model_selection import StratifiedKFold

MAPPINGS = {"skfold": StratifiedKFold}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def calculate_nr_bins_sturge_rule(X, column_name):

    N = X.shape[0]
    nr_bins = int(1 + np.log2(N))

    X["bins"] = pd.cut(X[column_name], bins=nr_bins, labels=False)
    print(X.head())

    return X


def load_wine():

    X = pd.read_csv(os.path.join(path, "red.csv"), sep=";")
    X["quality"] = X["quality"] - 3

    X = calculate_nr_bins_sturge_rule(X=X, column_name="total sulfur dioxide")

    return X


def report_cv(X):

    r = X.groupby(["fold"])["bins"].value_counts().unstack()

    # estamos en un caso de leave one out cv
    if len(r) == X.shape[0]:
        r = r.sum(axis=1)
        print(r)
        print(sum(r > 1))

    else:
        print(r)


def add_stratified_kfolds():

    X = load_wine()
    y = X["bins"]

    X["fold"] = -1

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=175)

    for split, (_, test_idx) in enumerate(skfold.split(X, y), start=1):
        X.loc[test_idx, "fold"] = split

    report_cv(X=X)

    return X


def main():
    add_stratified_kfolds()


if __name__ == "__main__":
    main()
