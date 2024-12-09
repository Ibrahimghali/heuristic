import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def particle_swarm_optimization_binary_with_local_search(func, N, D, Tmax, step):
    positions = np.random.randint(2, size=(N, D))
    velocities = np.random.uniform(-6, 6, (N, D))
    personal_best_positions = positions.copy()
    personal_best_costs = np.array([func(pos) for pos in positions])
    global_best_index = np.argmin(personal_best_costs)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_cost = personal_best_costs[global_best_index]
    iteration = 0
    results = []

    while iteration < Tmax:
        inertia_weight = 0.9 - 0.5 * iteration / Tmax
        r1 = np.random.uniform(0, 1, (N, D))
        r2 = np.random.uniform(0, 1, (N, D))
        cognitive = 1.5 * r1 * (personal_best_positions - positions)
        social = 1.5 * r2 * (global_best_position - positions)
        velocities = inertia_weight * velocities + cognitive + social
        probabilities = sigmoid(velocities)
        random_values = np.random.uniform(0, 1, (N, D))
        positions = (probabilities >= random_values).astype(int)
        
        # Local search on best particles
        num_local_search = max(1, int(0.3 * N))  # Apply local search to top 30% particles
        costs = np.array([func(pos) for pos in positions])
        best_particle_indices = np.argsort(costs)[:num_local_search]
        
        for idx in best_particle_indices:
            current_position = positions[idx].copy()
            current_cost = costs[idx]
            
            # Try multiple local search steps
            for _ in range(3):  # Number of local search attempts
                # Create a neighbor by flipping random bits
                neighbor = current_position.copy()
                num_bits_to_flip = np.random.randint(1, max(2, int(D * 0.1)))  # Flip up to 10% of bits
                bits_to_flip = np.random.choice(D, num_bits_to_flip, replace=False)
                
                for bit in bits_to_flip:
                    neighbor[bit] = 1 - neighbor[bit]
                
                neighbor_cost = func(neighbor)
                
                # Update if better solution found
                if neighbor_cost < current_cost:
                    current_position = neighbor.copy()
                    current_cost = neighbor_cost
                    positions[idx] = current_position
                    costs[idx] = current_cost
        
        # Update personal and global bests
        better_mask = costs < personal_best_costs
        personal_best_costs[better_mask] = costs[better_mask]
        personal_best_positions[better_mask] = positions[better_mask].copy()
        
        new_global_best_index = np.argmin(personal_best_costs)
        if personal_best_costs[new_global_best_index] < global_best_cost:
            global_best_cost = personal_best_costs[new_global_best_index]
            global_best_position = personal_best_positions[new_global_best_index].copy()
        
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)
        iteration += 1
    
    return results