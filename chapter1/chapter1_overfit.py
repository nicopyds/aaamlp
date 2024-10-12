#!/usr/bin/env python3

import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


def load_wine(path):

    X = pd.read_csv(os.path.join(path, "red.csv"), sep=";")

    return X


X = load_wine(path=path)
X = X.sample(frac=1)
X["quality"] = X["quality"] - 3

X_train = X.head(1000)
X_test = X.tail(599)

y_train = X_train.pop("quality")
y_test = X_test.pop("quality")


# type hinting


def training_loop(X_train, X_test, y_train, y_test):

    accuracy_train = [0.5]
    accuracy_test = [0.5]

    for max_depth_ in range(1, 25):

        model = DecisionTreeClassifier(max_depth=max_depth_)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train_ = accuracy_score(y_true=y_train, y_pred=y_train_pred)
        accuracy_test_ = accuracy_score(y_true=y_test, y_pred=y_test_pred)

        print(max_depth_, accuracy_train_, accuracy_test_)

        accuracy_train.append(accuracy_train_)
        accuracy_test.append(accuracy_test_)

    return accuracy_train, accuracy_test


def plot_scores(accuracy_train, accuracy_test):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots()

    x_ticks = [i + 1 for i in range(len(accuracy_train))]

    ax.plot(x_ticks, accuracy_train, label="train_score")
    ax.plot(x_ticks, accuracy_test, label="test_score")

    plt.legend()
    plt.show()


def main():

    accuracy_train, accuracy_test = training_loop(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    plot_scores(accuracy_train, accuracy_test)


if __name__ == "__main__":
    main()
