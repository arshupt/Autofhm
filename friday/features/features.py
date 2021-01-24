from featuretools import *
from friday.utils import *


class Features(object) :
    def __init__(
        self, 
        id, 
        target_entity,
        primitives,
        serial, 
        entities=None,
        dataframes=None, 
        props=None,
        relationships=None, 
        corr_threshold=None, 
    ) :
        self.id = id
        self.entities = entities
        self.dataframes = dataframes
        self.target_entity = target_entity
        self.primitives = primitives
        self.props = props
        self.relationships = relationships
        self.corr_threshold = corr_threshold
        self.serial = serial
        self.features = None
        self.feature_matrix = None

    def build(self) :
        entity_set = self.create_entity_set(self.entities, self.relationships)

    def create_entity_set(self, entities, relationships) :
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
                variable_types=types or None
            )

        def add_relationships(relationships) :
            for relationship in relationships :
                table1 = entity_set[relationship["table1"]["name"]][relationship["table1"]["column"]]
                table2 = entity_set[relationship["table2"]["name"]][relationship["table2"]["column"]]
                entity_set = entity_set.add_relationship(
                    Relationship(table1, table2)
                )
            return entity_set

        entity_set = add_relationships(self.relationships)
        
        return entity_set

