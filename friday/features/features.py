from featuretools import *
from utils import *
import numpy as np

class Features(object) :
    def __init__(
        self, 
        id, 
        target_entity,
        primitives=None,
        serial=False, 
        entities=None,
        dataframes=None, 
        props=None,
        relationships=None, 
        split_ratio=0.2,
        corr_threshold=0.7, 
    ) :
        self.id = id
        self.entities = entities
        self.target_entity = target_entity
        self.dataframes = dataframes
        self.target_entity = target_entity
        self.primitives = primitives
        self.props = props
        self.relationships = relationships
        self.corr_threshold = corr_threshold
        self.split_ratio = split_ratio
        self.serial = serial

    def build(self) :
        """
        creats entities and performs DFS. gives :feature_matrix: 

        :feature_matrix: -  the final matrix with desired values
        """
        entity_set = self.create_entity_set(self.entities, self.relationships)

        feature_matrix, _ = dfs(
            entity_set=entity_set,
            target_entity=self.target_entity, 
            **self.props)

        target_entity_index = None
        for entity in self.entities :
            if(entity["id"]==self.target_entity) :
                target_entity_index = entity["index"]
                break
        
        feature_matrix = self.reorder(feature_matrix, target_entity_index)

        feature_matrix = self.process_feature_matrix(feature_matrix, self.corr_threshold)

        return feature_matrix


    def create_entity_set(self, entities, relationships, normalize_entity_id=None) :
        """
        converts the dataframe entities to feature tool Entity class object
        """
        entity_set = EntitySet(id=self.id)

        for entity in entities :
            entity_id = entity["id"]
            dataframe = self.dataframes.get(entity_id.upper())
            index = entity["index"]
            types = entity["variable_types"]
            variable_types = {}
            for k,v in types :
                variable_types[k] = DEFAULT_VARIABLE_TYPES[v]
            entity_set = entity_set.entity_from_dataframe(
                entity_id=entity_id,
                dataframe=dataframe,
                index=index,
                variable_types=variable_types or None
            )

        def entity_relationships(relationships) :
            ''' Add realationship from the config file into the coresponding entity. 
            :relationships: - It the input array of realtionsships of eath enity

            This attach the entity object with 
            '''
            for relationship in relationships :
                table1 = entity_set[relationship["table1"]["name"]][relationship["table1"]["column"]]
                table2 = entity_set[relationship["table2"]["name"]][relationship["table2"]["column"]]
                entity_set = entity_set.add_relationship(
                    Relationship(table1, table2)
                )
            return entity_set

        entity_set = entity_relationships(self.relationships)

        if(len(entities)==1) :
            if(normalize_entity_id==None) :
                raise Exception("Normalize entity id not found")
            entity_set = entity_set.normalize_entity(
                base_entity_id=entities[0]["id"], 
                new_entity_id=normalize_entity_id, 
                index=normalize_entity_id)
        
        return entity_set

    def reorder(self, feature_matrix, reindex_id) :
        feature_matrix = feature_matrix.reindex(index=reindex_id)
        feature_matrix = feature_matrix.reset_index()
        
        return feature_matrix

    def process_feature_matrix(self, feature_matrix, corr_threshold=None) :
        """ uses the correlation matrix and the threshhold to select (by setting as :True:) those columns that have correlation above :corr_threshold:
        """
        corr_matrix = feature_matrix.corr().abs() 

        upper = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1)
            .astype(np.bool)
        )

        collinear_features = [ column for column in upper.columns if any(upper[column]>corr_threshold)]
        feature_matrix = feature_matrix.drop(columns = collinear_features)

        return feature_matrix
