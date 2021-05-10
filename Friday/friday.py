import os
import warnings
import random
import pickle

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from Friday.utils.metrics import metrics
from Friday.utils.parse_util import parse_json, parse_xml, parse_yaml
from Friday.utils.utils import cv_score
from Friday.feature.features import Features
from Friday.ga.ga import GeneticAlgo

class Friday :

    def __init__(self, 
                config=None, 
                random_state=42) :

        self._config = config
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None
        self._model = None
        self.feature = None

        self.random_state = random_state
        self.feature_config, self.model_config, self.training_config = self._handle_config()  
        self.cv = 5 if 'cv' not in self.training_config else self.training_config['cv']
        self.n_jobs = -1 if 'n_jobs' not in self.training_config else self.training_config['n_jobs']
        self.scoring_function = None if 'scoring_function' not in self.training_config else self.training_config['scoring_function']
        self.classification = True if self.training_config['mode']=='classification' else False

    def _handle_config(self) :

        feature_config = None
        model_config = None
        training_config = None
        config = self._config

        if isinstance(config, dict):
            feature_config = config['feature']
            model_config = config['model']
            training_config = config['train_mode']

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
                training_config = config_dict['training_mode']

            elif index == 1:
                config_dict, err = parse_xml(config)
                if err is not None:
                    raise Exception(err.args)
                feature_config = config_dict['feature']
                model_config = config_dict['model']
                training_config = config_dict['training_mode']

            elif index == 2:
                config_dict, err = parse_json(config)
                if err is not None:
                    raise Exception(err.args)
                feature_config = config_dict['feature']
                model_config = config_dict['model']
                training_config = config_dict['training_mode']
        
        return feature_config, model_config, training_config

    def get_features(self) :  

        id = self.feature_config['entity_set']['id']
        target_entity = self.feature_config['entity_set']['target_entity']
        primitives = self.feature_config['primitives']
        entities = self.feature_config['entities']
        relationships = self.feature_config['relationship']
        corr_threshold = 0.9 if 'corr_threshold' not in self.feature_config else self.feature_config['corr_threshold']
        scale = None if 'scale' not in self.feature_config else self.feature_config['scale']
        transform = None if 'transform' not in self.feature_config else self.feature_config['transform']
        test_size = 0.2 if 'test_size' not in self.feature_config else self.feature_config['test_size']
        
        self.feature = Features(
            id,
            target_entity,
            primitives=primitives,
            entities=entities,
            relationships=relationships,
            corr_threshold=corr_threshold,
            scale=scale,
            transform=transform,
            test_size=test_size,
            n_jobs=self.n_jobs,
            random_state=self.random_state,

        )

        self.X_train, self.X_test, self.y_train, self.y_test = self.feature.build()

    def get_test_data(self, n=10) :

        return pd.merge(self.X_test, self.y_test, left_index=True, right_index=True).sample(n)

    def _get_optimised_pipeline(self) :

        config_dict = self.model_config if self.model_config else None

        training_config = self.training_config

        gen = 20 if 'gen' not in training_config else training_config['gen']
        population = 50 if 'population' not in training_config else training_config['population']
        offspring =population if 'offspring' not in training_config else training_config['offspring']
        mutation_rate = 0.8 if 'mutation_rate' not in training_config else training_config['mutation_rate']
        crossover_rate = 0.2 if 'crossover_rate' not in training_config else training_config['crossover_rate']

        if self.classification :
            model = DecisionTreeClassifier(max_depth=5)
        else :
            model = DecisionTreeRegressor(max_depth=5)

        try :
            model.fit(self.X_train, self.y_train)
        except :
            raise ValueError('Input Data is not in a valid form')

        ga = GeneticAlgo(
            generations=gen, 
            population_size=population, 
            offspring_size=offspring, 
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            cv=self.cv, 
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            config_dict=config_dict,
            classification=self.classification,
            scoring_function=self.scoring_function
            )

        return ga.optimise(self.X_train, self.y_train)

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

    def test(self, X_test=None, y_test=None):

        if self.scoring_function is None :
            if self.classification :
                scorer = metrics['accuracy']
            else :
                scorer = metrics['explained_variance']
        else :
            scorer = metrics[self.scoring_function]

        if X_test is not None and y_test is not None :
            self.X_test = X_test
            self.y_test = y_test
        elif self.X_test is None and self.y_test is None :
            raise ValueError('No data is provided')
        y_pred = self.predict(self.X_test)
        score = scorer(self.y_test, y_pred)
        return score

    def save_model(self, filename, path) :

        model_name = filename + ".sav"
        try :
            pickle.dump(self._model, open(os.path.join(path, model_name), 'wb'))
        except Exception as e: 
            raise Exception(e)

        print(f"Model saved to the folder {path} with name {model_name}")

    def load_model(self, path) :

        try :
            model = pickle.load(open(path, 'rb'))
        except  Exception as e :
            raise Exception(e)
        self._model = model
        print('Done!')