import numpy as np
import random
from deap import tools, gp
from inspect import isclass
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from collections import defaultdict

def mutIndividual(ind, pset):

    if(len(ind)<=2) : return ind
    index = np.random.randint(2, len(ind))
    node = ind[index]
    node_str = node.name
    i = 0
    while node_str == node.name and i<10 :
        node = np.random.choice(pset.terminals[node.ret])
        i+=1
    if node.name!=node_str :
        ind[index] = node
    return ind

def cxOnePoint(ind1, ind2):
    
    if str(ind1) == str(ind2) : return ind1, ind2
    indexList = []
    for i, node in enumerate(zip(ind1[1:], ind2[1:]), 1) :
        node1, node2  = node
        if node1!=node2 :
            indexList.append(i)
    if len(indexList) != 0 :
        index = random.choice(indexList)
        ind1[index], ind2[index] = ind2[index], ind1[index]
    return ind1, ind2


def selectIndividual(population) :

    types = defaultdict(list)
    typesName = set()

    for ind in population: 
        name = ind[0].name
        types[name].append(ind)
        typesName.add(name)
    
    selectedType = np.random.choice(list(typesName))
    ind1 = random.choice(types[selectedType])
    ind2 = random.choice(types[selectedType])
    return ind1, ind2

def varAnd(population, toolbox, lambda_, cxpb, mutpb):
    
    offspring = []
    
    for _ in range(lambda_):
        op_choice = np.random.random()
        if op_choice < cxpb:
            ind1, ind2 = selectIndividual(population)
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
    offspring = random.sample(offspring, lambda_)
    return offspring

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, console, halloffame=None):

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.evaluate(invalid_ind)
    
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    console.start_pb_loading()
    for gen in range(1, ngen + 1):
        console.update_current(f"Generation [{gen}/{ngen}]")
        offspring = varAnd(population, toolbox, lambda_, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(offspring)

        population = toolbox.select(population + offspring, mu)
        console.advance(100/ngen)
        console.log(f"Generation {gen} Completed.")
        
    return population