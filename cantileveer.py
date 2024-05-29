import numpy as np
import random

def objective_function(x):
    return 0.6224 * np.sum(x)

def constraint_function(x):
    return (61 / x[0]**3) + (27 / x[1]**3) + (19 / x[2]**3) + (7 / x[3]**3) + (1 / x[4]**3) - 1

# SGO Parameters
population_size = 50
max_iterations = 100
learning_rate = 0.5

# Initialization
population = np.random.uniform(low=0.01, high=100, size=(population_size, 5))

for iteration in range(max_iterations):
    # Evaluation
    fitness = np.array([objective_function(x) for x in population])
    valid_indices = [i for i, x in enumerate(population) if constraint_function(x) >= 0]

    if len(valid_indices) > 0:
        best_index = valid_indices[np.argmin(fitness[valid_indices])]
        best_solution = population[best_index]

        # SGO Update
        for i in range(population_size):
            if i != best_index:
                r1 = random.uniform(0, 1)  # Random number for improving phase
                r2 = random.uniform(0, 1)  # Random number for acquiring phase
                j = random.randint(0, population_size - 1)  # Random individual

                # Improving phase
                population[i] = population[i] + learning_rate * r1 * (best_solution - population[i])

                # Acquiring phase
                population[i] = population[i] + learning_rate * r2 * (population[j] - population[i])

                # Boundary check
                population[i] = np.clip(population[i], 0.01, 100)

# Final evaluation
fitness = np.array([objective_function(x) for x in population])
valid_indices = [i for i, x in enumerate(population) if constraint_function(x) >= 0]
best_index = valid_indices[np.argmin(fitness[valid_indices])]

print("Optimal lengths:", population[best_index])
print("Minimum value of the objective function:", fitness[best_index])