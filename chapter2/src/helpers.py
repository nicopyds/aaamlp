#! /usr/bin/env python

import os
import sys

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

file_path = os.path.abspath(__file__)
src_path = os.path.join(os.path.dirname(file_path), "ml_metrics")

sys.path.insert(0, src_path)

from ml_metrics import (
    custom_accuracy,
    custom_accuracy_2,
    custom_tp,
    custom_tn,
    custom_fp,
    custom_fn,
    custom_precision,
    custom_recall,
    custom_pr_from_proba,
)

dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(file_path)), "data", "user_behavior_dataset.csv"
)

MANUFACTURER_MAPPING_DICT = {
    "Google": 1,
    "OnePlus": 2,
    "Samsung": 3,
    "Xiaomi": 4,
    "iPhone": 5,
}


def load_df(path):

    return pd.read_csv(path)


def clean_column_names(df):

    # aplicamos el map para limpiar los espacios
    cols = list(map(lambda col: col.replace(" ", ""), df.columns))

    # aplicamos el map para quitar los (
    cols = list(map(lambda col: col.split("(")[0] if "(" in col else col, cols))

    df.columns = cols

    return df


def extract_mobile_manufacturer(value):

    return value.split(" ")[0]


def clean_df(path):

    df = load_df(path=path)

    df = (
        df.pipe(clean_column_names)
        .set_index("UserID")
        .assign(
            OperatingSystem=lambda df: (df["OperatingSystem"] == "iOS") * 1,
            Gender=lambda df: (df["Gender"] == "Male") * 1,
            DeviceModel=lambda df: df["DeviceModel"].apply(extract_mobile_manufacturer),
        )
        .assign(DeviceModel=lambda df: df["DeviceModel"].map(MANUFACTURER_MAPPING_DICT))
    )

    return df


def get_train_test_datasets(X: pd.DataFrame, target_column: str = "Gender"):

    y = X.pop(target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=175
    )

    return X_train, X_test, y_train, y_test


def train_model(model, *datasets: pd.DataFrame):

    X_train, X_test, y_train, y_test = datasets
    model.fit(X_train, y_train)

    return model


def compare_metrics(y_true, y_pred, y_pred_proba):

    # fmt: off

    print("-" * 50)
    print(f"scikit-learn accuracy {round(metrics.accuracy_score(y_true=y_true, y_pred=y_pred), 2)}")
    print(f"custom metrics accuracy {custom_accuracy(y_true=y_true, y_pred=y_pred)}")
    print(f"custom metrics accuracy v2 {custom_accuracy_2(y_true=y_true, y_pred=y_pred)}")

    print("-" * 50)
    print(metrics.confusion_matrix(y_true, y_pred))

    print(f"custom metrics train tp {custom_tp(y_true=y_true, y_pred=y_pred)}")
    print(f"custom metrics train tn {custom_tn(y_true=y_true, y_pred=y_pred)}")
    print(f"custom metrics train fp {custom_fp(y_true=y_true, y_pred=y_pred)}")
    print(f"custom metrics train fn {custom_fn(y_true=y_true, y_pred=y_pred)}")


    print("-" * 50)
    print(f"scikit-learn precision {round(metrics.precision_score(y_true=y_true, y_pred=y_pred), 2)}")
    print(f"custom metrics train precision {custom_precision(y_true=y_true, y_pred=y_pred)}")

    print("-" * 50)
    print(f"scikit-learn recall {round(metrics.recall_score(y_true=y_true, y_pred=y_pred), 2)}")
    print(f"custom metrics train recall {custom_recall(y_true=y_true, y_pred=y_pred)}")

    precision, recall = custom_pr_from_proba(y_true, y_pred_proba=y_pred_proba)

    plt.figure(figsize=(7, 7))
    plt.plot(recall, precision)
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.show()


# fmt: on


def main(path):

    X = clean_df(path=path)
    X_train, X_test, y_train, y_test = get_train_test_datasets(X=X)

    model = DecisionTreeClassifier(max_depth=2)
    model = train_model(model, X_train, X_test, y_train, y_test)

    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]

    compare_metrics(
        y_true=y_train, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba
    )


if __name__ == "__main__":
    main(path=dataset_path)
