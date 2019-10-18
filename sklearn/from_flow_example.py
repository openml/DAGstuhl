#!/usr/bin/env python3
#
# Expected to be used with the flow pipeline of the example pipeline based on:
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

import json
import sys

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from sklearn_flow import from_flow

np.random.seed(0)

# Read data from Titanic dataset.
titanic_url = 'https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv'

data = pd.read_csv(titanic_url)

X = data.drop('survived', axis=1)
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = from_flow(json.load(sys.stdin))

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
