#!/usr/bin/env python3

import pandas as pd
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y = True, as_frame=True)


def main():
    print(type(X), type(y))


if __name__ == "__main__":
    main()
