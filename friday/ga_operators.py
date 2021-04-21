import numpy as np
from deap import tools, gp
from inspect import isclass
from .operator_utils import set_sample_weight
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from collections import defaultdict
import warnings
import threading


def mutNodeReplacement(ind, pset):

    index = np.random.randint(0, len(ind))
    node = ind[index]
    subtree = ind.searchSubtree(index)

    if node.arity == 0: 
        term = np.random.choice(pset.terminals[node.ret])
        ind[index] = term if not isclass(term) else term()
    else:  
        rindex = None

        if index + 1 < len(ind):
            for i, temp in enumerate(ind[index+1:], index+ 1):
                if isinstance(temp, gp.Primitive) and temp.ret in temp.args:
                    rindex = i
        primitives = pset.primitives[node.ret]

        if len(primitives) != 0:
            new_node = np.random.choice(primitives)
            new_subtree = [None] * len(new_node.args)

            if rindex:
                rnode = ind[rindex]
                rsubtree = ind.searchSubtree(rindex)
                position = np.random.choice([i for i, a in enumerate(new_node.args) if a == rnode.ret])
            else:
                position = None

            for i, arg_type in enumerate(new_node.args):

                if i != position:
                    term = np.random.choice(pset.terminals[arg_type])

                    if isclass(term):
                        term = term()
                    new_subtree[i] = term

            if rindex:
                new_subtree[position:position + 1] = ind[rsubtree]
            new_subtree.insert(0, new_node)
            ind[subtree] = new_subtree

    return ind


def cxOnePoint(ind1, ind2):

    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for index, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(index)
    common_types = []

    for index, node in enumerate(ind2[1:], 1):

        if node.ret in types1 and not node.ret in types2:
            common_types.append(node.ret)
        types2[node.ret].append(index)

    if len(common_types) > 0:
        ret_type = np.random.choice(common_types)

        index1 = np.random.choice(types1[ret_type])
        index2 = np.random.choice(types2[ret_type])

        subtree1 = ind1.searchSubtree(index1)
        subtree2 = ind2.searchSubtree(index2)
        ind1[subtree1], ind2[subtree2] = ind2[subtree2], ind1[subtree1]

    return ind1, ind2


def varAnd(population, toolbox, lambda_, cxpb, mutpb):
    
    offspring = []

    for _ in range(lambda_):
        op_choice = np.random.random()

        if op_choice < cxpb:           
            idxs = np.random.randint(0, len(population),size=2)
            ind1, ind2 = toolbox.clone(population[idxs[0]]), toolbox.clone(population[idxs[1]])
            ind_str = str(ind1)
            num_loop = 0

            while ind_str == str(ind1) and num_loop < 50 : 
                ind1, ind2 = toolbox.mate(ind1, ind2)
                num_loop += 1

            if ind_str != str(ind1): 
                del ind1.fitness.values
            offspring.append(ind1) 

        op_choice = np.random.random()

        if op_choice < mutpb:  
            idx = np.random.randint(0, len(population))
            ind = toolbox.clone(population[idx])
            ind_str = str(ind)
            num_loop = 0

            while ind_str == str(ind) and num_loop < 50 :
                ind = toolbox.mutate(ind)
                num_loop += 1

            if ind_str != str(ind): 
                del ind.fitness.values
            offspring.append(ind)
        else: 
            idx = np.random.randint(0, len(population))
            offspring.append(toolbox.clone(population[idx]))

    return np.random.sample(offspring, lambda_)

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, halloffame=None):

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.evaluate(invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    for gen in range(1, ngen + 1):

        offspring = varAnd(population, toolbox, lambda_, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)

        population = toolbox.select(population + offspring, mu)

    return population