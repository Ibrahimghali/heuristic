import numpy as np
import matplotlib.pyplot as plt
import time



# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def particle_swarm_optimization_binary_condition(func, N, D, Tmax, step):
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
        # Calcul dynamique des coefficients d'accélération selon D
        if D == 28:  # Problème de petite dimension
            cognitive_coeff = 3.30
            social_coeff = 3.30
        elif D == 60:  # Problème de dimension moyenne
            cognitive_coeff = 7.46
            social_coeff = 7.46
        else:  # Problème de grande dimension
            cognitive_coeff = 13.31
            social_coeff = 13.31

        # Mise à jour de l'inertie
        inertia_weight = 0.9 - 0.5 * iteration / Tmax

        # Générer des valeurs aléatoires pour les composants cognitifs et sociaux
        r1 = np.random.uniform(0, 1, (N, D))
        r2 = np.random.uniform(0, 1, (N, D))

        # Calcul des vitesses en incluant les coefficients ajustés
        cognitive = cognitive_coeff * r1 * (personal_best_positions - positions)
        social = social_coeff * r2 * (global_best_position - positions)
        velocities = inertia_weight * velocities + cognitive + social

        # Mise à jour des positions en appliquant la fonction sigmoïde
        probabilities = sigmoid(velocities)
        random_values = np.random.uniform(0, 1, (N, D))
        positions = (probabilities >= random_values).astype(int)

        # Calcul des coûts et mise à jour des meilleures positions
        costs = np.array([func(pos) for pos in positions])
        better_mask = costs < personal_best_costs
        personal_best_costs[better_mask] = costs[better_mask]
        personal_best_positions[better_mask] = positions[better_mask].copy()

        # Mise à jour de la meilleure solution globale
        new_global_best_index = np.argmin(personal_best_costs)
        if personal_best_costs[new_global_best_index] < global_best_cost:
            global_best_cost = personal_best_costs[new_global_best_index]
            global_best_position = personal_best_positions[new_global_best_index].copy()

        # Stocker le résultat à intervalles réguliers
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)

        iteration += 1

    return results
