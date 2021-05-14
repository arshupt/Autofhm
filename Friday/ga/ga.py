import warnings
import random
from functools import partial
from inspect import isclass
from multiprocessing import cpu_count

from joblib import Parallel, delayed

import deap
import numpy as np
import pandas as pd
from deap import gp, tools, creator, base
from copy import copy


from .config_model import config_classifier, config_regressor
from .ga_operators import cxOnePoint, mutIndividual, eaMuPlusLambda
from Friday.utils.utils import cv_score, Output_Array, pareto_eq, findOperatorClass, expr_to_tree, generate_pipeline_code


class GeneticAlgo :

    def __init__(self,generations=20, population_size=50, offspring_size=None,
                 mutation_rate=0.8, crossover_rate=0.2,
                 cv=5, n_jobs=-1,random_state=None, 
                 config_dict=None, classification=True, scoring_function=None) :
        
        self._pareto_front = None
        self._optimized_pipeline = None
        self._fitted_pipeline = None
        self._pop = None

        self.population_size = population_size
        self.generations = generations
        self.offspring_size = population_size if offspring_size is None else offspring_size
        self.classification = classification   
        if config_dict is None: 
            self.config_dict = config_classifier if self.classification else config_regressor
        else :
            self.config_dict = config_dict

        self.mutpb = mutation_rate
        self.cxpb = crossover_rate
        self.random_state = random_state
        if scoring_function is not None :
            self.scoring_function = scoring_function
        else :
            if self.classification :
                self.scoring_function = 'accuracy'
            else :
                self.scoring_function = 'explained_variance'
        self.cv = cv
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs > cpu_count():
            print(f"n_jobs given is more than the number of cores available, settinr n_jobs to {cpu_count()}")
            self.n_jobs = cpu_count()
        else :
            self.n_jobs = n_jobs

        self.operators = []
        self.arguments = []
        for key in sorted(self.config_dict.keys()):
            op_class, arg_types = findOperatorClass(key, self.config_dict[key])
            if op_class:
                self.operators.append(op_class)
                self.arguments += arg_types

        self._evaluated_individuals = {}
        self.operators_context = {
            'copy': copy
        }
        self._setup_pset()
        self._setup_toolbox()
        
    def _setup_pset(self): 
        
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        self._pset = gp.PrimitiveSetTyped('MAIN', [np.ndarray], Output_Array)
        self._pset.renameArguments(ARG0='input_matrix')

        for op in self.operators:

            if op.root:
                p_types = (op.parameter_types()[0], Output_Array)
                self._pset.addPrimitive(op, *p_types)

            self._pset.addPrimitive(op, *op.parameter_types())
            for key in sorted(op.import_hash.keys()):
                module_list = op.import_hash[key][0]

                exec('from {} import {}'.format(key, module_list))
                for var in op.import_hash[key]:
                    self.operators_context[var] = eval(var)

               
        for _type in self.arguments:

            for val in list(_type.values):
                terminal_name = _type.__name__ + "=" + str(val)
                self._pset.addTerminal(val, _type, name=terminal_name)

    def _setup_toolbox(self):
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti, statistics=dict)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', self._gen_grow, pset=self._pset, min_=1, max_=3)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('compile', self._compile_to_sklearn)
        self._toolbox.register('select', tools.selNSGA2)
        self._toolbox.register('mate', self._mate_operator)
        self._toolbox.register('expr_mut', self._gen_grow, min_=1, max_=1)
        self._toolbox.register('mutate', self._random_mutation_operator)

    def optimise(self, features, classes, sample_weight=None):

        features = features.astype(np.float64)

        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self._toolbox.register('evaluate', self._evaluate_individuals, features=features, classes=classes)

        pop = self._toolbox.population(n=self.population_size)

        self._pareto_front = tools.ParetoFront(similar=pareto_eq)

    
        total_evals = self.offspring_size * self.generations + self.population_size

        try:
            pop = eaMuPlusLambda(population=pop, toolbox=self._toolbox,
                mu=self.population_size, lambda_=self.offspring_size,
                cxpb=self.cxpb, mutpb=self.mutpb,
                ngen=self.generations, halloffame=self._pareto_front)

        except (KeyboardInterrupt, SystemExit):
            print('Keyboard interrupt!. Will use the best pipeline so far')

        finally:
            if self._pareto_front:
                top_score = -float('inf')
                for pipeline, pipeline_scores in zip(self._pareto_front.items, reversed(self._pareto_front.keys)):
                    if pipeline_scores.wvalues[1] > top_score:
                        self._optimized_pipeline = pipeline
                        top_score = pipeline_scores.wvalues[1]

                if not self._optimized_pipeline:
                    print('No model is optimized. Please re run the program after checking the config')
                    return None
                else:
                    self._fitted_pipeline = self._toolbox.compile(expr=self._optimized_pipeline)

                    self._fitted_pipeline.fit(features, classes)
                    return self._fitted_pipeline
            
    
    def _mate_operator(self, ind1, ind2):
        return cxOnePoint(ind1, ind2)

    def _random_mutation_operator(self, individual):
        mutation_technique = partial(mutIndividual, pset=self._pset)
        return mutation_technique(individual)

    def _gen_grow(self, pset, min_, max_, type_=None):
        return self._generate(pset, type_)

    def _generate(self, pset, type_=None):
        if type_ is None:
            type_ = pset.ret
        expr = []
        try:
            prim = np.random.choice(pset.primitives[type_])
        except IndexError:
            raise IndexError("Some error occured while adding primitives") 
        expr.append(prim)
        for arg in prim.args:
            try:
                term = np.random.choice(pset.terminals[arg])
            except IndexError:
                raise IndexError("Some error occured while adding terminal primitives")
            if isclass(term):
                term = term()
            expr.append(term)
        return expr

    def _evaluate_individuals(self, individuals, features, classes, sample_weight = None):
        
        fitnesses_dict = {}
        eval_individuals_str = []
        sklearn_pipeline_list = []
        operator_count_list = []
        test_idx_list = []
        for indidx, individual in enumerate(individuals):
            individual = individuals[indidx]
            individual_str = str(individual)
            if individual_str in self._evaluated_individuals:
                fitnesses_dict[indidx] = self._evaluated_individuals[individual_str]
       
            else:
                try:
                    sklearn_pipeline = self._toolbox.compile(expr=individual)
                    operator_count = 0
                    for i in range(len(individual)):
                        node = individual[i]
                        if ((type(node) is deap.gp.Terminal) or
                             type(node) is deap.gp.Primitive ):
                            continue
                        operator_count += 1
                except Exception:
                    fitnesses_dict[indidx] = (5000., -float('inf'))
                    continue
                eval_individuals_str.append(individual_str)
                operator_count_list.append(operator_count)
                sklearn_pipeline_list.append(sklearn_pipeline)
                test_idx_list.append(indidx)

        resulting_score_list = []
        for chunk_idx in range(0, len(sklearn_pipeline_list),self.n_jobs*4):
            parallel = Parallel(n_jobs=self.n_jobs, verbose=0, pre_dispatch='2*n_jobs')
            tmp_result_score = parallel(delayed(cv_score)(sklearn_pipeline, features, classes,
                                        self.cv, self.scoring_function, self.random_state)
                                        for sklearn_pipeline in sklearn_pipeline_list[chunk_idx:chunk_idx+self.n_jobs*4])
            for val in tmp_result_score:
                if val == 'Timeout':
                    resulting_score_list.append(-float('inf'))
                else:
                    resulting_score_list.append(val)

        for resulting_score, operator_count, individual_str, test_idx in zip(resulting_score_list, operator_count_list, eval_individuals_str, test_idx_list):
            if type(resulting_score) in [float, np.float64, np.float32]:
                self._evaluated_individuals[individual_str] = (max(1, operator_count), resulting_score)
                fitnesses_dict[test_idx] = self._evaluated_individuals[individual_str]
            else:
                raise ValueError('Scoring function does not return a float.')

        fitnesses_ordered = []
        for key in sorted(fitnesses_dict.keys()):
            fitnesses_ordered.append(fitnesses_dict[key])
        return fitnesses_ordered

    def _set_param_recursive(self, pipeline_steps, parameter, value):

        for (_, obj) in pipeline_steps:
            recursive_attrs = ['steps', 'transformer_list', 'estimators']
            for attr in recursive_attrs:
                if hasattr(obj, attr):
                    self._set_param_recursive(getattr(obj, attr), parameter, value)
            if hasattr(obj, 'estimator'):  
                est = getattr(obj, 'estimator')
                if hasattr(est, parameter):
                    setattr(est, parameter, value)
            if hasattr(obj, parameter):
                setattr(obj, parameter, value)

    def _compile_to_sklearn(self, expr):
        sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(expr, self._pset), self.operators)
        model = eval(sklearn_pipeline_str, self.operators_context)
        return model
