#!/usr/bin/env python3
#
# This example pipeline is based on:
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

import json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

import jsonschema
import requests

from sklearn_flow import to_flow

numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='lbfgs')),
])

flow = to_flow(pipeline)

jsonschema.validate(flow, requests.get('https://openml.github.io/flow2/schemas/v0/pipeline.json').json())

print(json.dumps(flow, allow_nan=False, ensure_ascii=False, indent=2))
