from featuretools import dfs, EntitySet, Relationship
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from default_types import DEFAULT_VARIABLE_TYPES
from utils import index_util
from unit_dfs import unit_dfs
from scaling import max_abs_scaling, min_max_scaling, mean_normalize, standardization, robust_scaling, DEFAULT_SCALING
import numpy as np
import pandas as pd


class Features(object):
    def __init__(
            self,
            id,
            target_entity,
            primitives=None,
            serial=0,
            entities=None,
            props=None,
            relationships=None,
            test_size=0.2,
            corr_threshold=0.7,
    ):
        self.entity_set = None
        self.feature_matrix = None
        self.id = id
        self.entities = entities
        self.target = None
        self.target_entity = target_entity
        self.primitives = primitives,
        self.relationships = relationships
        self.corr_threshold = corr_threshold
        self.variables = dict()
        self.serial = serial == 1
        self.scale_mode = "min_max_scaling"
        self.test_size = test_size

    def build(self):
        self.entity_set, self.target, self.variables = self.create_entity_set(self.entities, self.relationships)

        if(len(self.entities)==1) :
            self.feature_matrix, numerical_columns = unit_dfs(self.entities)
            for column in numerical_columns :
                self.variables["numerical"].add(column)
        else :
            self.feature_matrix, _ = dfs(
                    entityset=self.entity_set,
                    target_entity=self.target_entity,
                    max_depth = 2, 
                    verbose = 3
                )

        correlated_columns = self.process_feature_matrix(self.feature_matrix, self.corr_threshold)
        self.feature_matrix.drop(correlated_columns, axis=1)
        df = self.scale_values(self.scale_mode)
        self.feature_matrix = self.reorder(self.feature_matrix, df)

        X_train, X_test, y_train, y_test = self.test_train_split(df)
        return X_train, X_test, y_train, y_test

    def create_entity_set(self, entities, relationships):
        entity_set = EntitySet(id=self.id)
        target = None
        variable_types = {
            "numerical": set(),
            "categorical": set(),
            "boolean": set()
        }
        for entity in entities:
            entity_id = entity["id"]
            index = entity["index"]
            types = entity["variable_types"]
            v_types = {}
            for k, v in types.items():
                v_types[k] = DEFAULT_VARIABLE_TYPES[v]
                variable_types[v].add(k)
            dataframe = pd.read_csv(entity["dataframe_path"])
            #dataframe = dataframe.reset_index(inplace=True)
            if "target_column" in entity :
                target = dataframe[entity["target_column"]]
                dataframe = dataframe.drop(entity["target_column"], axis=1)
            entity_set = entity_set.entity_from_dataframe(
                entity_id=entity_id,
                dataframe=dataframe,
                index=index,
                variable_types=v_types or None
            )

        def entity_relationships(relationships):
            if relationships is None:
                return
            for relationship in relationships:
                table1 = entity_set[relationship["table1"]["name"]][relationship["table1"]["column"]]
                table2 = entity_set[relationship["table2"]["name"]][relationship["table2"]["column"]]
                entity_set = entity_set.add_relationship(
                    Relationship(table1, table2)
                )

        entity_relationships(self.relationships)
        return entity_set, target, variable_types

    def reorder(self, feature_matrix, df):
        feature_matrix = index_util(feature_matrix)
        df = index_util(df)
        if(feature_matrix is None) :
            self.feature_matrix = df
            return df
        feature_matrix.merge(df)
        return feature_matrix

    def process_feature_matrix(self, feature_matrix, corr_threshold=None):
        corr = feature_matrix.corr()
        col_corr = set()
        for i in range(len(corr.columns)) :
            for j in range(i) :
                if(abs(corr.iloc[i,j])>.85) :
                    col_corr.add(corr.columns[i])
        return col_corr

    def scale_values(self, scale_mode) :

        scale_function = DEFAULT_SCALING[scale_mode]
        columns = self.variables["numerical"]

        df = scale_function(self.feature_matrix, columns)
        df = pd.DataFrame(df, columns=columns)
        self.feature_matrix.drop(columns, axis=1)

        return df


    def test_train_split(self, df) : 

        X_train, X_test, y_train, y_test = train_test_split(df, self.target, test_size=.2)
        
        return X_train, X_test, y_train, y_test