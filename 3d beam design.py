import numpy as np
import random

def objective_function(x):
    return 0.8 * x[0] + x[1] + 0.8 * x[2]

def constraint_function(x):
    return (11.248 * 10**-3) / x[0]**3 + (3.5399 * 10**-3) / x[1]**3 + (0.384 * 10**-3) / x[2]**3 - 0.05

# SGO Parameters
population_size = 50
max_iterations = 100
learning_rate = 0.5  # You might need to tune this

# Initialization
population = np.random.uniform(low=0.1, high=0.9, size=(population_size, 3))

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

                # Boundary check (Keep values within bounds)
                population[i] = np.clip(population[i], 0.1, 0.9)

# Final evaluation
fitness = np.array([objective_function(x) for x in population])
valid_indices = [i for i, x in enumerate(population) if constraint_function(x) >= 0]
best_index = valid_indices[np.argmin(fitness[valid_indices])]

print("Optimal values (SGO):", population[best_index])
print("Minimum value of the objective function (SGO):", fitness[best_index])
