import random
from deap import base, creator, tools

def initialize_population(pop_size, genome_length, indpb=0.5):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -5, 5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, genome_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    return toolbox.population(n=pop_size)

def evaluate_population(population, eval_function):
    fitnesses = map(eval_function, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

def select_best(population, k):
    return tools.selBest(population, k)

def crossover(parent1, parent2, cxpb=0.5):
    tools.cxBlend(parent1, parent2, alpha=cxpb)
    return parent1, parent2

def mutate(individual, mutpb=0.2):
    tools.mutGaussian(individual, mu=0, sigma=1, indpb=mutpb)
    return individual


