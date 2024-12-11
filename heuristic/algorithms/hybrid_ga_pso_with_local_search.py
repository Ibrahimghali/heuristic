import numpy as np

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

def hybrid_ga_pso_with_local_search(func, N, D, Tmax, step):
    """
    Hybrid Genetic Algorithm and Particle Swarm Optimization with Local Search.

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
    velocities = np.random.uniform(-6, 6, (N, D))  # Continuous velocities
    fitnesses = np.array([func(pos) for pos in positions])  # Evaluate initial fitnesses

    # Initialize GA variables
    mutation_rate = 0.2  # Initial mutation rate
    mutation_decay = 0.99  # Decay mutation rate over time
    
    # Initialize PSO variables
    personal_best_positions = positions.copy()  # Personal best positions
    personal_best_costs = fitnesses.copy()  # Personal best costs
    global_best_index = np.argmin(personal_best_costs)  # Index of global best
    global_best_position = personal_best_positions[global_best_index].copy()  # Global best position
    global_best_cost = personal_best_costs[global_best_index]  # Global best cost

    results = []  # List to store results
    iteration = 0  # Iteration counter
    stagnation_counter = 0  # Stagnation counter
    stagnation_threshold = 10  # Replace stagnant individuals every 10 iterations

    while iteration < Tmax:
        if iteration % 2 == 0:
            # Perform GA step
            for i in range(N):
                parent1, parent2 = np.random.choice(N, 2, replace=False)  # Select parents
                cross_point = np.random.randint(1, D - 1)  # Crossover point
                child = np.hstack((positions[parent1, :cross_point], positions[parent2, cross_point:]))  # Crossover
                
                # Apply mutation with adaptive rate
                mutation_mask = np.random.rand(D) < mutation_rate
                child = np.where(mutation_mask, 1 - child, child)  # Mutate child

                child_fitness = func(child)  # Evaluate child fitness
                if child_fitness < fitnesses[i]:  # Replace if better
                    positions[i] = child
                    fitnesses[i] = child_fitness

            mutation_rate *= mutation_decay  # Decay mutation rate

        else:
            # Perform PSO step
            inertia_weight = 0.9 - 0.5 * iteration / Tmax  # Inertia weight
            cognitive_weight = 1.5 + (iteration / Tmax)  # Increase cognitive weight
            social_weight = 1.5 - (iteration / Tmax)  # Decrease social weight
            r1 = np.random.uniform(0, 1, (N, D))  # Cognitive random factors
            r2 = np.random.uniform(0, 1, (N, D))  # Social random factors
            cognitive = cognitive_weight * r1 * (personal_best_positions - positions)  # Cognitive component
            social = social_weight * r2 * (global_best_position - positions)  # Social component
            velocities = inertia_weight * velocities + cognitive + social  # Update velocities
            probabilities = sigmoid(velocities)  # Sigmoid transformation
            positions = (probabilities > np.random.uniform(0, 1, (N, D))).astype(int)  # Update positions

            # Evaluate new positions
            costs = np.array([func(pos) for pos in positions])  # Evaluate fitnesses
            better_mask = costs < personal_best_costs  # Identify improvements
            personal_best_costs[better_mask] = costs[better_mask]  # Update personal best costs
            personal_best_positions[better_mask] = positions[better_mask].copy()  # Update personal best positions

            new_global_best_index = np.argmin(personal_best_costs)  # Find new global best
            if personal_best_costs[new_global_best_index] < global_best_cost:
                global_best_cost = personal_best_costs[new_global_best_index]
                global_best_position = personal_best_positions[new_global_best_index].copy()

            fitnesses = costs  # Update fitnesses

        # Handle stagnation: replace worst solutions periodically
        if stagnation_counter >= stagnation_threshold:
            worst_indices = np.argsort(fitnesses)[-int(0.2 * N):]  # Identify worst individuals
            for idx in worst_indices:
                positions[idx] = np.random.randint(2, size=D)  # Reinitialize positions
                fitnesses[idx] = func(positions[idx])  # Reevaluate fitnesses
            stagnation_counter = 0  # Reset stagnation counter

        # Enhanced local search on top solutions
        num_local_search = max(1, int(0.3 * N))  # Number of individuals for local search
        best_indices = np.argsort(fitnesses)[:num_local_search]  # Identify best individuals
        for idx in best_indices:
            individual = positions[idx]
            for _ in range(10):  # Perform more local search steps
                neighbor = individual.copy()
                flip_indices = np.random.choice(D, int(max(2, D * 0.1)), replace=False)  # Flip bits
                neighbor[flip_indices] = 1 - neighbor[flip_indices]
                neighbor_cost = func(neighbor)
                if neighbor_cost < fitnesses[idx]:  # Replace if better
                    individual = neighbor
                    fitnesses[idx] = neighbor_cost

            positions[idx] = individual  # Update positions

        # Track global best
        stagnation_counter += 1  # Increment stagnation counter
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)  # Record result

        iteration += 1  # Increment iteration

    return results  # Return results
