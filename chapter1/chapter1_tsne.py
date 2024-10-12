#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn import set_config

set_config(transform_output="pandas")

from sklearn import datasets
from sklearn import manifold

import seaborn as sns
import matplotlib.pyplot as plt

data = datasets.fetch_openml("mnist_784", version=1, return_X_y=True)


X, y = data
y = y.astype(int)

one_image = X.iloc[0, :]
one_image_reshaped = one_image.array.reshape(28, 28)

label_one_image = y.loc[0]

# lectura recomendable sobre el tsne
# https://www.datacamp.com/tutorial/introduction-t-sne?dc_referrer=https%3A%2F%2Fwww.google.com%2F

# Xt = manifold.TSNE(2, random_state=175).fit_transform(X)
# Xt["y"] = y

Xt = manifold.TSNE(2, random_state=175).fit_transform(X.iloc[:1000, :])
Xt["y"] = y.iloc[:3000]


def plot_one_image(image, title):
    plt.imshow(image, cmap="gray")
    plt.title(str(title))
    plt.show()


def plot_tnse_image(X):
    grid = sns.FacetGrid(X, hue="y")
    grid.map(plt.scatter, "tsne0", "tsne1").add_legend()
    plt.show()


def main():
    plot_one_image(image=one_image_reshaped, title=label_one_image)
    plot_tnse_image(X=Xt)


if __name__ == "__main__":
    main()
