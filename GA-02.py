import random
import numpy as np

# تابع Rastrigin برای مینیمیزه کردن
def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def initialize_population(population_size, dimension, x_min, x_max):
    population = []
    for _ in range(population_size):
        chromosome = [round(random.uniform(x_min, x_max), 2) for _ in range(dimension)]
        population.append(chromosome)
    return population

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1))
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(chromosome, mutation_rate, x_min, x_max):
    mutated_chromosome = []
    for gene in chromosome:
        if random.random() < mutation_rate:
            new_value = gene + random.uniform(-0.5, 0.5)
            mutated_chromosome.append(max(x_min, min(x_max, new_value)))
        else:
            mutated_chromosome.append(gene)
    return mutated_chromosome

def genetic_algorithm(population_size, dimension, x_min, x_max, generations, mutation_rate):
    population = initialize_population(population_size, dimension, x_min, x_max)
    for _ in range(generations):
        next_generation = []
        for _ in range(population_size):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, x_min, x_max)
            next_generation.append(child)
        population = next_generation

    best_solution = min(population, key=rastrigin_function)
    return best_solution

# استفاده از الگوریتم ژنتیک برای بهینه‌سازی تابع Rastrigin
population_size = 100
dimension = 2  # تعداد متغیرها
x_min = -5.12
x_max = 5.12
generations = 100
mutation_rate = 0.15
best_solution = genetic_algorithm(population_size, dimension, x_min, x_max, generations, mutation_rate)
print("Best solution:", best_solution)
print("Best fitness:", rastrigin_function(best_solution))