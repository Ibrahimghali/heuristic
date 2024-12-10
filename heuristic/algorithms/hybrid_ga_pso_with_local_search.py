import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hybrid_ga_pso_with_local_search(func, N, D, Tmax, step):
    # Initialize population
    positions = np.random.randint(2, size=(N, D))
    velocities = np.random.uniform(-6, 6, (N, D))
    fitnesses = np.array([func(pos) for pos in positions])

    # Initialize GA variables
    mutation_rate = 0.2  # Initial mutation rate
    mutation_decay = 0.99  # Decay mutation rate over time
    
    # Initialize PSO variables
    personal_best_positions = positions.copy()
    personal_best_costs = fitnesses.copy()
    global_best_index = np.argmin(personal_best_costs)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_cost = personal_best_costs[global_best_index]

    results = []
    iteration = 0
    stagnation_counter = 0
    stagnation_threshold = 10  # Replace stagnant individuals every 10 iterations

    while iteration < Tmax:
        if iteration % 2 == 0:
            # Perform GA step
            for i in range(N):
                parent1, parent2 = np.random.choice(N, 2, replace=False)
                cross_point = np.random.randint(1, D - 1)
                child = np.hstack((positions[parent1, :cross_point], positions[parent2, cross_point:]))
                
                # Apply mutation with adaptive rate
                mutation_mask = np.random.rand(D) < mutation_rate
                child = np.where(mutation_mask, 1 - child, child)

                child_fitness = func(child)
                if child_fitness < fitnesses[i]:
                    positions[i] = child
                    fitnesses[i] = child_fitness

            mutation_rate *= mutation_decay  # Decay mutation rate

        else:
            # Perform PSO step
            inertia_weight = 0.9 - 0.5 * iteration / Tmax
            cognitive_weight = 1.5 + (iteration / Tmax)  # Increase cognitive weight
            social_weight = 1.5 - (iteration / Tmax)  # Decrease social weight
            r1 = np.random.uniform(0, 1, (N, D))
            r2 = np.random.uniform(0, 1, (N, D))
            cognitive = cognitive_weight * r1 * (personal_best_positions - positions)
            social = social_weight * r2 * (global_best_position - positions)
            velocities = inertia_weight * velocities + cognitive + social
            probabilities = sigmoid(velocities)
            positions = (probabilities > np.random.uniform(0, 1, (N, D))).astype(int)

            # Evaluate new positions
            costs = np.array([func(pos) for pos in positions])
            better_mask = costs < personal_best_costs
            personal_best_costs[better_mask] = costs[better_mask]
            personal_best_positions[better_mask] = positions[better_mask].copy()

            new_global_best_index = np.argmin(personal_best_costs)
            if personal_best_costs[new_global_best_index] < global_best_cost:
                global_best_cost = personal_best_costs[new_global_best_index]
                global_best_position = personal_best_positions[new_global_best_index].copy()

            fitnesses = costs

        # Handle stagnation: replace worst solutions periodically
        if stagnation_counter >= stagnation_threshold:
            worst_indices = np.argsort(fitnesses)[-int(0.2 * N):]
            for idx in worst_indices:
                positions[idx] = np.random.randint(2, size=D)
                fitnesses[idx] = func(positions[idx])
            stagnation_counter = 0

        # Enhanced local search on top solutions
        num_local_search = max(1, int(0.3 * N))
        best_indices = np.argsort(fitnesses)[:num_local_search]
        for idx in best_indices:
            individual = positions[idx]
            for _ in range(10):  # Perform more local search steps
                neighbor = individual.copy()
                flip_indices = np.random.choice(D, int(max(2, D * 0.1)), replace=False)
                neighbor[flip_indices] = 1 - neighbor[flip_indices]
                neighbor_cost = func(neighbor)
                if neighbor_cost < fitnesses[idx]:
                    individual = neighbor
                    fitnesses[idx] = neighbor_cost

            positions[idx] = individual

        # Track global best
        stagnation_counter += 1
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)

        iteration += 1

    return results
