import os
import matplotlib.pyplot as plt
from genetic_algorithm import genetic_algorithm
import time

def sphere(individual):
    return sum(x**2 for x in individual),

pop_sizes = [50, 100, 200]
mutation_rates = [0.1, 0.2, 0.3]
crossover_rates = [0.5, 0.7, 0.9]
generations = 5
genome_length = 10
niche_sizes = [5, 10, 20]
radius = 0.5
niche_capacity = 5

def run_experiment(pop_size, diversity_maintenance, **kwargs):
    start_time = time.time()
    best_individual = genetic_algorithm(
        pop_size=pop_size,
        genome_length=genome_length,
        eval_function=sphere,
        generations=generations,
        diversity_maintenance=diversity_maintenance,
        **kwargs
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    return best_individual.fitness.values[0], elapsed_time

def plot_results(x_values, clearing_results, crowding_results, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x_values, clearing_results, marker='o', label='Clearing')
    plt.plot(x_values, crowding_results, marker='o', label='Crowding')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f'img/{filename}')
    plt.close()

# Eksperymenty dla różnych rozmiarów populacji
pop_results_clearing = []
pop_results_crowding = []
pop_times_clearing = []
pop_times_crowding = []
for pop_size in pop_sizes:
    best_fit_clearing, time_clearing = run_experiment(pop_size, 'clearing', crossover_rate=0.7, mutation_rate=0.2, radius=radius, niche_capacity=niche_capacity)
    best_fit_crowding, time_crowding = run_experiment(pop_size, 'crowding', crossover_rate=0.7, mutation_rate=0.2, niche_size=5)
    pop_results_clearing.append(best_fit_clearing)
    pop_results_crowding.append(best_fit_crowding)
    pop_times_clearing.append(time_clearing)
    pop_times_crowding.append(time_crowding)

plot_results(
    pop_sizes, pop_results_clearing, pop_results_crowding,
    'Rozmiar populacji', 'Najlepsza wartość funkcji kosztu',
    'Wpływ rozmiaru populacji na najlepszą wartość funkcji kosztu',
    'population_size_comparison.png'
)

plot_results(
    pop_sizes, pop_times_clearing, pop_times_crowding,
    'Rozmiar populacji', 'Czas (sekundy)',
    'Wpływ rozmiaru populacji na czas wykonywania',
    'population_size_time_comparison.png'
)

# Eksperymenty dla różnych wskaźników mutacji
mut_results_clearing = []
mut_results_crowding = []
mut_times_clearing = []
mut_times_crowding = []
for mut_rate in mutation_rates:
    best_fit_clearing, time_clearing = run_experiment(100, 'clearing', crossover_rate=0.7, mutation_rate=mut_rate, radius=radius, niche_capacity=niche_capacity)
    best_fit_crowding, time_crowding = run_experiment(100, 'crowding', crossover_rate=0.7, mutation_rate=mut_rate, niche_size=5)
    mut_results_clearing.append(best_fit_clearing)
    mut_results_crowding.append(best_fit_crowding)
    mut_times_clearing.append(time_clearing)
    mut_times_crowding.append(time_crowding)

plot_results(
    mutation_rates, mut_results_clearing, mut_results_crowding,
    'Wskaźnik mutacji', 'Najlepsza wartość funkcji kosztu',
    'Wpływ wskaźnika mutacji na najlepszą wartość funkcji kosztu',
    'mutation_rate_comparison.png'
)

plot_results(
    mutation_rates, mut_times_clearing, mut_times_crowding,
    'Wskaźnik mutacji', 'Czas (sekundy)',
    'Wpływ wskaźnika mutacji na czas wykonywania',
    'mutation_rate_time_comparison.png'
)

# Eksperymenty dla różnych wskaźników krzyżowania
cross_results_clearing = []
cross_results_crowding = []
cross_times_clearing = []
cross_times_crowding = []
for cross_rate in crossover_rates:
    best_fit_clearing, time_clearing = run_experiment(100, 'clearing', crossover_rate=cross_rate, mutation_rate=0.2, radius=radius, niche_capacity=niche_capacity)
    best_fit_crowding, time_crowding = run_experiment(100, 'crowding', crossover_rate=cross_rate, mutation_rate=0.2, niche_size=5)
    cross_results_clearing.append(best_fit_clearing)
    cross_results_crowding.append(best_fit_crowding)
    cross_times_clearing.append(time_clearing)
    cross_times_crowding.append(time_crowding)

plot_results(
    crossover_rates, cross_results_clearing, cross_results_crowding,
    'Wskaźnik krzyżowania', 'Najlepsza wartość funkcji kosztu',
    'Wpływ wskaźnika krzyżowania na najlepszą wartość funkcji kosztu',
    'crossover_rate_comparison.png'
)

plot_results(
    crossover_rates, cross_times_clearing, cross_times_crowding,
    'Wskaźnik krzyżowania', 'Czas (sekundy)',
    'Wpływ wskaźnika krzyżowania na czas wykonywania',
    'crossover_rate_time_comparison.png'
)

# Eksperymenty dla różnych liczby nisz w populacji dla metody clearingu
niche_results_clearing = []
niche_times_clearing = []
for niche_size in niche_sizes:
    best_fit_clearing, time_clearing = run_experiment(100, 'clearing', crossover_rate=0.7, mutation_rate=0.2, radius=radius, niche_capacity=niche_size)
    niche_results_clearing.append(best_fit_clearing)
    niche_times_clearing.append(time_clearing)

plt.figure()
plt.plot(niche_sizes, niche_results_clearing, marker='o')
plt.xlabel('Liczba nisz/klastrów')
plt.ylabel('Najlepsza wartość funkcji kosztu')
plt.title('Wpływ liczby nisz/klastrów na najlepszą wartość funkcji kosztu (Clearing)')
plt.savefig('img/niche_size_clearing.png')
plt.close()

plt.figure()
plt.plot(niche_sizes, niche_times_clearing, marker='o')
plt.xlabel('Liczba nisz/klastrów')
plt.ylabel('Czas (sekundy)')
plt.title('Wpływ liczby nisz/klastrów na czas wykonywania (Clearing)')
plt.savefig('img/niche_size_time_clearing.png')
plt.close()

print("Wyniki eksperymentów:")
print("Wpływ rozmiaru populacji na najlepszą wartość funkcji kosztu (Clearing):", pop_results_clearing)
print("Wpływ rozmiaru populacji na najlepszą wartość funkcji kosztu (Crowding):", pop_results_crowding)
print("Wpływ rozmiaru populacji na czas wykonywania (Clearing):", pop_times_clearing)
print("Wpływ rozmiaru populacji na czas wykonywania (Crowding):", pop_times_crowding)
print("Wpływ wskaźnika mutacji na najlepszą wartość funkcji kosztu (Clearing):", mut_results_clearing)
print("Wpływ wskaźnika mutacji na najlepszą wartość funkcji kosztu (Crowding):", mut_results_crowding)
print("Wpływ wskaźnika mutacji na czas wykonywania (Clearing):", mut_times_clearing)
print("Wpływ wskaźnika mutacji na czas wykonywania (Crowding):", mut_times_crowding)
print("Wpływ wskaźnika krzyżowania na najlepszą wartość funkcji kosztu (Clearing):", cross_results_clearing)
print("Wpływ wskaźnika krzyżowania na najlepszą wartość funkcji kosztu (Crowding):", cross_results_crowding)
print("Wpływ wskaźnika krzyżowania na czas wykonywania (Clearing):", cross_times_clearing)
print("Wpływ wskaźnika krzyżowania na czas wykonywania (Crowding):", cross_times_crowding)
print("Wpływ liczby nisz/klastrów na najlepszą wartość funkcji kosztu (Clearing):", niche_results_clearing)
print("Wpływ liczby nisz/klastrów na czas wykonywania (Clearing):", niche_times_clearing)
