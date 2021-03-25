import pandas as pd
import numpy as np
from default_primitives import DEFAULT_PRIMITIVES
from featuretools.variable_types import Categorical, Numeric, Boolean, Discrete, Index, Id, Datetime, Timedelta, Text
from utils import create_dataframe_from_entity


def unit_dfs(entities, primitives=None) :
    
    assert entities, "None type received , expects List"

    primitive_list = dict()
    
    if primitives :
        primitive_list = primitives["transformation"]
    else :
        primitive_list = DEFAULT_PRIMITIVES


    dataframe, variable_types = create_dataframe_from_entity(entities[0])

    numerical_columns = list()
    categorical_columns = list()
    discrete_columns = list()
    boolean_columns = list()

    for k,v in variable_types.items() :
        
        if v=="numerical" :
            numerical_columns.append(k)
        elif v=="categorical" :
            categorical_columns.append(k)
        elif v=="boolean" :
            boolean_columns.append(k)
        elif v=="discrete" :
            discrete_columns.append(k)
        else :
            print("variable type {} is not implemented yet".format(v))

    generated_columns = []
    if(numerical_columns!=[]) :
        for key, primitive in primitive_list.items() :
            transformed_columns = primitive(dataframe, numerical_columns)
            for transformed_column in transformed_columns :
                dataframe[transformed_column[0]] = transformed_column[1]
                generated_columns.append(transformed_column[0])

    return dataframe, generated_columns