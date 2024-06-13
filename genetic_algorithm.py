import numpy as np
import random
from deap import base, creator, tools
from utils import initialize_population, evaluate_population, select_best, crossover, mutate
import copy

if not hasattr(creator, 'FitnessMin'):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def genetic_algorithm(pop_size, genome_length, eval_function, generations, diversity_maintenance, **kwargs):
    population = initialize_population(pop_size, genome_length)
    evaluate_population(population, eval_function)
    
    for gen in range(generations):
        offspring = tools.selTournament(population, len(population), tournsize=3)
        offspring = list(map(copy.deepcopy, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < kwargs.get('crossover_rate', 0.5):
                crossover(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < kwargs.get('mutation_rate', 0.2):
                mutate(mutant)
                del mutant.fitness.values
        
        evaluate_population(offspring, eval_function)
        
        if diversity_maintenance == "clearing":
            population.extend(offspring)
            population = _clearing(population, kwargs['radius'], kwargs['niche_capacity'])
        elif diversity_maintenance == "crowding":
            _crowding(population, offspring, kwargs['niche_size'])
    
    return select_best(population, 1)[0]

def _crowding(population, offspring, niche_size):
    for child in offspring:
        candidates = random.sample(population, niche_size)
        closest = min(candidates, key=lambda ind: np.linalg.norm(np.array(ind) - np.array(child)))
        if child.fitness.values >= closest.fitness.values:
            population.remove(closest)
            population.append(child)

def _clearing(population, radius, niche_capacity):
    niches = []
    for ind in population:
        in_niche = False
        for niche in niches:
            if np.linalg.norm(np.array(ind) - np.array(niche[0])) < radius:
                niche.append(ind)
                in_niche = True
                break
        if not in_niche:
            niches.append([ind])
    
    cleared_population = []
    for niche in niches:
        niche.sort(key=lambda x: x.fitness.values, reverse=True)
        cleared_population.extend(niche[:niche_capacity])
    
    return cleared_population
