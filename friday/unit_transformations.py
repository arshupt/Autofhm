import pandas as pd 
import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer


def log_transfom(dataframe, columns) :

    log_transfomer = FunctionTransformer(np.log, validate=True)

    transformed_columns = []
    for column in columns :
        new_values = log_transfomer.transform(dataframe[[column]])
        transformed_columns.append([column+"_log_transform",new_values])
    return transformed_columns


def sqrt_transform(dataframe, columns) :

    sqrt_transformer = FunctionTransformer(np.sqrt, validate=True)

    transformed_columns = []
    for column in columns :
        new_values = sqrt_transformer.transform(dataframe[[column]])
        transformed_columns.append([column+"_sqrt_transform",new_values])
    return transformed_columns


def reciprocal_transform(dataframe, columns) :

    reciprocal_transformer = FunctionTransformer(np.reciprocal, validate=True)

    transformed_columns = []
    for column in columns :
        new_values = reciprocal_transformer.transform(dataframe[[column]])
        transformed_columns.append([column+"_reciprocal_transform",new_values])
    return transformed_columns

def power_transform(dataframe, columns) :

    power_transformer = FunctionTransformer(lambda x: x**3, validate=True)

    transformed_columns = []
    for column in columns :
        new_values = power_transformer.transform(dataframe[[column]])
        transformed_columns.append([column+"_power_transform",new_values])
    return transformed_columns

