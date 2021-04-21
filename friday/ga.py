import warnings
import random
from inspect import isclass
from multiprocessing import cpu_count

from deap import gp, tools, creator, base
import deap
import numpy as np
import copy as copy

from utils import pareto_eq

class GeneticAlgo :

    def __init__(self,generations=100, population_size=100, offspring_size=None,
                 mutation_rate=0.9, crossover_rate=0.1,
                 scoring=None, cv=5, n_jobs=1,
                 max_time_mins=None, random_state=None, config_dict=None,) :
        self._pareto_front = None
        self._optimized_pipeline = None
        self._fitted_pipeline = None
        self._pop = None
        self.warm_start = warm_start
        self.population_size = population_size
        self.generations = generations
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.offspring_size = population_size

        self.operators = []
        self.arguments = []
        for key in sorted(self.config_dict.keys()):
            op_class, arg_types = findOperatorClass(key, self.config_dict[key])
            if op_class:
                self.operators.append(op_class)
                self.arguments += arg_types

        if not (max_time_mins is None):
            self.generations = 1000000

        self.mutpb = mutation_rate
        self.cxpb = crossover_rate

        if self.mutpb + self.cxpb > 1:
            raise ValueError('The sum of the crossover and mutation probabilities must be <= 1.0.')


        self._evaluated_individuals = {}

        self.random_state = random_state

        self.cv = cv

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

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
                module_list = ', '.join(sorted(op.import_hash[key]))

                exec('from {} import {}'.format(key, module_list))

                for var in op.import_hash[key]:
                    self.operators_context[var] = eval(var)

        self._pset.addPrimitive(CombineDFs(), [np.ndarray, np.ndarray], np.ndarray)

        for _type in self.arguments:
            type_values = list(_type.values)
            if 'nthread' not in _type.__name__:
                type_values += ['DEFAULT']

            for val in type_values:
                terminal_name = _type.__name__ + "=" + str(val)
                self._pset.addTerminal(val, _type, name=terminal_name)

        if self.verbosity > 2:
            print('{} operators have been imported by TPOT.'.format(len(self.operators)))

    def _setup_toolbox(self):
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', self._gen_grow, pset=self._pset, min_=1, max_=3)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('compile', self._compile_to_sklearn)
        self._toolbox.register('select', tools.selNSGA2)
        self._toolbox.register('mate', self._mate_operator)
        self._toolbox.register('expr_mut', self._gen_grow, min_=1, max_=4)
        self._toolbox.register('mutate', self._random_mutation_operator)

    def optimise(self, features, classes, sample_weight=None):

        features = features.astype(np.float64)

        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self._start_datetime = datetime.now()
        self._toolbox.register('evaluate', self._evaluate_individuals, features=features, classes=classes, sample_weight=sample_weight)
        pop = self._toolbox.population(n=self.population_size)

        self._pareto_front = tools.ParetoFront(similar=pareto_eq)

        if self.max_time_mins:
            total_evals = self.population_size
        else:
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
                    print('An error happended while optimising model')
                else:
                    self._fitted_pipeline = self._toolbox.compile(expr=self._optimized_pipeline)

                    return self._fitted_pipeline


    def _mate_operator(self, ind1, ind2):
        return cxOnePoint(ind1, ind2)

    def _random_mutation_operator(self, individual):
        mutation_techniques = [
            partial(gp.mutInsert, pset=self._pset),
            partial(mutNodeReplacement, pset=self._pset),
            partial(gp.mutShrink)
        ]
        return np.random.choice(mutation_techniques)(individual)

    def _gen_grow(self, pset, min_, max_, type_=None):
        def condition(height, depth, type_):
            return type_ not in [np.ndarray, Output_Array] or depth == height

        return self._generate(pset, min_, max_, condition, type_)

    def _generate(self, pset, min_, max_, condition, type_=None):
        if type_ is None:
            type_ = pset.ret
        expr = []
        height = np.random.randint(min_, max_)
        stack = [(0, type_)]
        while len(stack) != 0:
            depth, type_ = stack.pop()
            if condition(height, depth, type_):
                try:
                    term = np.random.choice(pset.terminals[type_])
                except IndexError:
                    raise IndexError("Some error occured while adding terminal primitives")
                if isclass(term):
                    term = term()
                expr.append(term)
            else:
                try:
                    prim = np.random.choice(pset.primitives[type_])
                except IndexError:
                    raise IndexError("Some error occured while adding primitives") 
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth+1, arg))

        return expr