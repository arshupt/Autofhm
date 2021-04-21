from friday.features import Feature
from friday.utils.parse_util import parse_json, parse_xml, parse_yaml
import pandas as pd
import os
import warnings

import os
import warnings
import pandas as pd
import numpy as np
import random

from features import Features
from parse_util import parse_json, parse_xml, parse_yaml

class Friday :

    def __init__(self, 
                config, 
                max_time=None, 
                max_score=None, 
                test_size=0.2, 
                corr_threshold=0.7) :

        self._config = config
        self._max_time = max_time
        self._max_score = max_score
        self._test_size = test_size
        self._corr_threshold = corr_threshold
        self._features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None
        self._model = None

        self._feature_config, self._model_config, self._train_mode_config = self._handle_config()  

    def _handle_config(self) :

        feature_config = None
        model_config = None
        train_mode_config = None
        config = self.config

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
        
        return feature_config, model_config, train_mode_config

    def _get_features(self) :  

        feature_config = self.feature_config

        id = feature_config['entity_set']['id']
        target_entity = feature_config['entity_set']['target_entity']
        primitives = feature_config['primitives']
        entities = feature_config['entities']
        relationships = feature_config['relationship']

        feature = Features(
            id,
            target_entity,
            primitives=primitives,
            entities=entities,
            relationships=relationships,
        )

        self.X_train, self.X_test, self.y_train, self.y_test = feature.build()


    def _get_optimised_pipeline(self) :

        if self.train_mode_config['mode']=='Classification' :
            model = DecisionTreeClassifier(max_depth=5)
        else :
            model = DecisionTreeRegressor(max_depth=5)

        try :
            model.fit(self.X_train, self.y_train)
        except :
            raise ValueError('Input Data is not in a valid form')

        ga = GeneticAlgo()

        model = ga.optimise(self.X_train, self.y_train)


    def fit(self, X_train=None, y_train=None):
        
        if X_train is not None and y_train is not None :
            self.X_train = X_train
            self.y_train = y_train
        elif self.X_train is None and self.y_train is None :
            raise ValueError('No data is provided')

        if self.random_state is not None :
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self._model = self._get_optimised_pipeline()


    def predict(self, features):

        if not self._model:
            raise RuntimeError('Optimised model not found, please call the fit() first or fit_predict() first')

        features = features.astype(np.float64)

        return self._model.predict(features)

    def fit_predict(self, features, classes):
  
        self.fit(features, classes)

        output =  self.predict(features)
        return output