import numpy as np

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

def improved_hybrid_ga_pso(func, N, D, Tmax, step):
    """
    Improved Hybrid Genetic Algorithm and Particle Swarm Optimization.

    Parameters:
    func (callable): The objective function to minimize.
    N (int): The population size.
    D (int): The dimensionality of the problem.
    Tmax (int): The maximum number of iterations.
    step (int): The step interval for recording results.

    Returns:
    list: The list of best fitness values recorded at each step interval.
    """
    # Initialize population
    positions = np.random.randint(2, size=(N, D))  # Binary positions
    velocities = np.random.uniform(-4, 4, (N, D))  # Continuous velocities
    fitnesses = np.array([func(pos) for pos in positions])  # Evaluate initial fitnesses

    # Initialize GA variables
    mutation_rate = 0.2
    crossover_rate = 0.8

    # Initialize PSO variables
    personal_best_positions = positions.copy()
    personal_best_costs = fitnesses.copy()
    global_best_index = np.argmin(personal_best_costs)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_cost = personal_best_costs[global_best_index]

    results = []
    iteration = 0

    while iteration < Tmax:
        # Alternate between PSO and GA
        if iteration % 2 == 0:
            # Perform PSO step
            inertia_weight = 0.9 - 0.4 * (iteration / Tmax)
            cognitive_weight = 2.0
            social_weight = 2.0
            r1 = np.random.uniform(0, 1, (N, D))
            r2 = np.random.uniform(0, 1, (N, D))

            # Update velocities and positions
            cognitive = cognitive_weight * r1 * (personal_best_positions - positions)
            social = social_weight * r2 * (global_best_position - positions)
            velocities = inertia_weight * velocities + cognitive + social
            velocities = np.clip(velocities, -4, 4)  # Clip velocities to prevent explosion
            probabilities = sigmoid(velocities)
            positions = (probabilities > np.random.uniform(0, 1, (N, D))).astype(int)

            # Evaluate fitnesses
            fitnesses = np.array([func(pos) for pos in positions])
            better_mask = fitnesses < personal_best_costs
            personal_best_costs[better_mask] = fitnesses[better_mask]
            personal_best_positions[better_mask] = positions[better_mask].copy()

            # Update global best
            global_best_index = np.argmin(personal_best_costs)
            global_best_cost = personal_best_costs[global_best_index]
            global_best_position = personal_best_positions[global_best_index].copy()
        else:
            # Perform GA step
            for i in range(N):
                if np.random.rand() < crossover_rate:
                    # Select parents and perform crossover
                    parent1, parent2 = np.random.choice(N, 2, replace=False)
                    cross_point = np.random.randint(1, D - 1)
                    child = np.hstack((positions[parent1, :cross_point], positions[parent2, cross_point:]))
                else:
                    # No crossover; child is a clone
                    child = positions[i].copy()

                # Apply mutation
                mutation_mask = np.random.rand(D) < mutation_rate
                child = np.where(mutation_mask, 1 - child, child)

                # Evaluate fitness and update if better
                child_fitness = func(child)
                if child_fitness < fitnesses[i]:
                    positions[i] = child
                    fitnesses[i] = child_fitness

            # Update mutation rate dynamically
            mutation_rate *= 0.98

        # Track global best during GA
        current_best_index = np.argmin(fitnesses)
        if fitnesses[current_best_index] < global_best_cost:
            global_best_cost = fitnesses[current_best_index]
            global_best_position = positions[current_best_index].copy()

        # Record results at intervals
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)

        iteration += 1

    return results
