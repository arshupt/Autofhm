from features import Features
from parse_util import parse_json, parse_xml, parse_yaml
import pandas as pd
import numpy as np
import os
import warnings


def fit(config=None):
    feature_config = None
    model_config = None
    train_mode_config = None
    serial = 1
    if isinstance(config, dict):
        feature_config = config['feature']
        model_config = config['model']
        train_mode_config = config['train_mode']

    elif isinstance(config, str):
        FOMRAT = ['yaml', 'xml', 'json']
        file_format = config.split('.')[-1]

        if file_format not in FOMRAT:
            raise Exception('The given path is invalid or the file format is not supported')

        index = FOMRAT.index(file_format)

        if index == 0:
            config_dict, err = parse_yaml(config)
            if err is not None:
                raise Exception(err.args)
            feature_config = config_dict['feature']
            model_config = config_dict['model']
            train_mode_config = config_dict['train_mode']

        elif index == 1:
            config_dict, err = parse_xml(config)
            if err is not None:
                raise Exception(err.args)
            feature_config = config_dict['feature']
            model_config = config_dict['model']
            train_mode_config = config_dict['train_mode']

        elif index == 2:
            config_dict, err = parse_json(config)
            if err:
                raise Exception(err.args)
            feature_config = config_dict['feature']
            model_config = config_dict['model']
            train_mode_config = config_dict['train_mode']

    id = feature_config['entity_set']['id']
    target_entity = feature_config['entity_set']['target_entity']
    primitives = feature_config['primitives']
    # serial = train_mode_config['serial']
    entities = feature_config['entities']
    relationships = None
    test_size = 0.2
    corr_threshold = 0.7
    feature = Features(
        id,
        target_entity,
        primitives=primitives,
        serial=serial,
        entities=entities,
        relationships=relationships,
        test_size=test_size,
        corr_threshold=corr_threshold
    )

    dataframe = feature.build()
    print(dataframe)

    

fit("D:\\arshu\\Documents\\Main Project\\Code\\AutoML\\Friday\\config.json")

