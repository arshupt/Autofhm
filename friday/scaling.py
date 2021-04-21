import pandas as pd
import numpy as np
import sklearn.preprocessing as sp


def mean_normalize(self, dataframe, columns) :

    mean = dataframe[columns].mean(axis=0)
    max_min = dataframe[columns].max(axis=0) - dataframe[columns].min(axis=0)
    
    df = (dataframe[columns] - mean) / max_min
    return df


def standardization(self, dataframe, columns) :

    df = dataframe[columns]
    scaler = sp.StandardScaler()
    scaler.fit(df)
    
    df = scaler.transform(df)
    return df


def robust_scaling(dataframe, columns) :

    df = dataframe[columns]
    scaler = sp.RobustScaler()
    scaler.fit(df)
    
    df = scaler.transform(df)
    return df


def min_max_scaling(dataframe, columns) :

    df = dataframe[columns]
    scaler = sp.MinMaxScaler()
    scaler.fit(df)
    
    df = scaler.transform(df)
    return df


def max_abs_scaling(dataframe, columns) :

    df = dataframe[columns]
    scaler = sp.MaxAbsScaler()
    scaler.fit(df)
    
    df = scaler.transform(df)
    return df


DEFAULT_SCALING = {
    "standardization" : standardization,
    "robust_scaling" : robust_scaling,
    "mean_normalize" : mean_normalize,
    "min_max_scaling" : min_max_scaling,
    "max_abs_scaling" : max_abs_scaling
}
