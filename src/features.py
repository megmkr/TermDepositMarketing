import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OrdinalEncoder

BINARY_COLUMNS = {
    'y': {'no': False, 'yes': True},
    'default': {'no': False, 'yes': True},
    'housing': {'no': False, 'yes': True},
    'loan': {'no': False, 'yes': True},
}

QUARTER_MAP = {
    'jan': 1, 'feb': 1, 'mar': 1,
    'apr': 2, 'may': 2, 'jun': 2,
    'jul': 3, 'aug': 3, 'sep': 3,
    'oct': 4, 'nov': 4, 'dec': 4,
}

SEASON_MAP = {
    'dec': 1, 'jan': 1, 'feb': 1,
    'mar': 2, 'apr': 2, 'may': 2,
    'jun': 3, 'jul': 3, 'aug': 3,
    'sep': 4, 'oct': 4, 'nov': 4,
}

UNKNOWN_COLUMNS = ['job', 'education', 'contact']


def map_binary_columns(df):
    df = df.copy()
    for column, mapping in BINARY_COLUMNS.items():
        df[column] = df[column].map(mapping)
    return df


def engineer_features(df):
    df = df.copy()
    df['quarter'] = df['month'].map(QUARTER_MAP)
    df['season'] = df['month'].map(SEASON_MAP)
    df['march'] = df['month'] == 'mar'
    df['campaign_squared'] = df['campaign'] ** 2
    df['many_contacts'] = df['campaign'] >= 10
    df['long_call'] = df['duration'] >= 1000
    df['duration_log'] = np.log1p(df['duration'])
    return df


def drop_unknown_rows(df, columns=UNKNOWN_COLUMNS):
    for column in columns:
        df = df[df[column] != 'unknown']
    return df


def encode_categorical(df):
    df = df.copy()
    category_columns = df.select_dtypes(include='object').columns
    df[category_columns] = OrdinalEncoder().fit_transform(df[category_columns])
    return df


def select_features(df, k=10):
    df = map_binary_columns(df)
    df = engineer_features(df)
    df = drop_unknown_rows(df)
    df = encode_categorical(df)

    X = df.drop(columns='y')
    y = df['y']

    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    X = X[X.columns[selector.get_support()]]

    return X, y
