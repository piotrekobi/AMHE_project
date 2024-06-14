import matplotlib.pyplot as plt
import time
import numpy as np
from genetic_algorithm import genetic_algorithm
import os

def sphere(individual):
    return sum(x**2 for x in individual),

def rastrigin(individual):
    return 10 * len(individual) + sum((x**2 - 10 * np.cos(2 * np.pi * x)) for x in individual),

def rosenbrock(individual):
    return sum(100 * (individual[i+1] - individual[i]**2)**2 + (individual[i] - 1)**2 for i in range(len(individual)-1)),

def griewank(individual):
    part1 = sum(x**2 for x in individual) / 4000
    part2 = np.prod([np.cos(x / np.sqrt(i+1)) for i, x in enumerate(individual)])
    return part1 - part2 + 1,

functions = [sphere, rastrigin, rosenbrock, griewank]
function_names = ["sphere", "rastrigin", "rosenbrock", "griewank"]

pop_size = 100
mutation_rate = 0.2
crossover_rate = 0.7
generations = 100
dimensions = [10, 30, 50]
niche_size = 5
radius = 0.5
niche_capacity = 5

def run_experiment(pop_size, genome_length, eval_function, diversity_maintenance, **kwargs):
    convergence = []
    best_fits = []
    times = []
    for _ in range(10):
        start_time = time.time()
        best_individual, conv = genetic_algorithm(
            pop_size=pop_size,
            genome_length=genome_length,
            eval_function=eval_function,
            generations=generations,
            diversity_maintenance=diversity_maintenance,
            convergence=convergence,
            **kwargs
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        best_fits.append(best_individual.fitness.values[0])
        times.append(elapsed_time)
    avg_best_fit = np.mean(best_fits)
    avg_time = np.mean(times)
    return avg_best_fit, avg_time, convergence

def plot_convergence(convergence, title, filename):
    plt.figure()
    for dim, conv in convergence.items():
        plt.plot(range(generations), conv, label=f'Dimension {dim}')
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness')
    plt.title(title)
    plt.legend()
    plt.savefig(f'img/{filename}')
    plt.close()

for function, function_name in zip(functions, function_names):
    convergence_clearing = {}
    convergence_crowding = {}
    for genome_length in dimensions:
        print(f"\nEksperymenty dla funkcji {function_name} z wymiarowością {genome_length}:")
        
        avg_best_fit_clearing, avg_time_clearing, conv_clearing = run_experiment(pop_size, genome_length, function, 'clearing', crossover_rate=crossover_rate, mutation_rate=mutation_rate, radius=radius, niche_capacity=niche_capacity)
        convergence_clearing[genome_length] = conv_clearing
        print(f"Clearing - Best Fitness: {avg_best_fit_clearing}, Time: {avg_time_clearing}")
        
        avg_best_fit_crowding, avg_time_crowding, conv_crowding = run_experiment(pop_size, genome_length, function, 'crowding', crossover_rate=crossover_rate, mutation_rate=mutation_rate, niche_size=niche_size)
        convergence_crowding[genome_length] = conv_crowding
        print(f"Crowding - Best Fitness: {avg_best_fit_crowding}, Time: {avg_time_crowding}")

    plot_convergence(convergence_clearing, f'Convergence for {function_name} with Clearing', f'convergence_clearing_{function_name}.png')
    plot_convergence(convergence_crowding, f'Convergence for {function_name} with Crowding', f'convergence_crowding_{function_name}.png')
