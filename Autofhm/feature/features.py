import numpy as np
import pandas as pd
from multiprocessing import cpu_count
import warnings

import featuretools as ft
from featuretools import dfs, EntitySet, Relationship, encode_features
from sklearn.model_selection import train_test_split

from .default_types import DEFAULT_VARIABLE_TYPES
from .scaling import DEFAULT_SCALINGS
from .transforms import DEFAULT_TRANSFORMS


class Features(object):
    def __init__(
        self,
        id,
        target_entity,
        entities,
        relationships=None,
        primitives=None,
        corr_threshold=0.8,
        scale=None,
        transforms=None,
        n_jobs=-1,
        test_size=0.2,
        random_state=30
    ):
        self.entity_set = None
        self.feature_matrix = None
        self.id = id
        self.entities = entities
        self.target = None
        self.target_entity = target_entity
        self.primitives = primitives
        self.relationships = relationships
        self.corr_threshold = corr_threshold
        self.variables = dict()
        self.columns = set()
        self.transforms= transforms
        self.scale_mode = scale
        self.n_jobs = n_jobs if n_jobs!=-1 else cpu_count()
        self.test_size = test_size
        self.random_state = random_state

    def build(self):

        self.entity_set, self.target, self.variables = self.create_entity_set(self.entities, self.relationships)
        
        if len(self.entities)==1 :

            primitives = self.primitives if self.primitives else ['add_numeric','multiply_numeric']

            feature_matrix, columns = dfs(entityset=self.entity_set, target_entity=self.target_entity, trans_primitives=primitives)
        else :
            feature_matrix, columns = dfs(
                    entityset=self.entity_set,
                    target_entity=self.target_entity,
                    trans_primitives=self.primitives
                )

        feature_matrix, columns = encode_features(feature_matrix, columns)

        correlated_columns = self.process_feature_matrix(feature_matrix, self.corr_threshold)

        self.feature_matrix = feature_matrix.drop(columns=correlated_columns)
        
        # self.feature_matrix = self.reorder(feature_matrix)

        return train_test_split(self.feature_matrix, self.target, shuffle=True, random_state=self.random_state, test_size=self.test_size)

    def create_entity_set(self, entities, relationships):
        entity_set = EntitySet(id=self.id)
        target = None
        variable_types = {
            "numerical": set(),
            "categorical": set(),
            "boolean": set(),
            "discrete":set(),
            "index":set(),
            "id":set(),
            "datetime":set(),
            "timedelta":set(),
            "text":set()
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
            with warnings.catch_warnings(): 
                warnings.filterwarnings('ignore')
                entity_set = entity_set.entity_from_dataframe(
                    entity_id=entity_id,
                    dataframe=dataframe,
                    index=index,
                    variable_types=v_types or None
                )

        def entity_relationships(relationships):
            nonlocal entity_set
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

    def reorder(self, feature_matrix):

        feature_matrix = feature_matrix.sort_index()
        return feature_matrix

    def process_feature_matrix(self, df, corr_threshold=None):
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        col_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        return col_corr

    def scale(self, df, cols, mode) :

        fnc = DEFAULT_SCALINGS[mode]

        df = fnc.transform(df, cols)

        return df

    def transform(self, df, cols, mode) :

        fnc = DEFAULT_TRANSFORMS[mode]

        df = fnc(df, cols)

        return df





