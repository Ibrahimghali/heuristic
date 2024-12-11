from decimal import Decimal

import numpy as np


import numpy as np

def cuckoo_search_binary(func, N, D, Tmax, step, pa=0.25, alpha=0.01):
    # Initialisation des positions des nids (solutions)
    positions = np.random.randint(2, size=(N, D))
    # Calcul des coûts (fonction objectif) pour chaque solution
    costs = np.array([func(pos) for pos in positions])

    # Meilleure solution globale
    global_best_index = np.argmin(costs)
    global_best_position = positions[global_best_index].copy()
    global_best_cost = costs[global_best_index]

    # Initialisation des résultats
    results = []

    iteration = 0
    while iteration < Tmax:
        # Générer des nids (solutions) par perturbation
        new_positions = positions.copy()
        for i in range(N):
            step_size = alpha * np.random.randn(D)
            new_positions[i] = (positions[i] + step_size) % 2  # Modifie la solution de manière aléatoire

        # Calcul des nouveaux coûts
        new_costs = np.array([func(pos) for pos in new_positions])

        # Remplacer les nids moins performants par les meilleurs
        better_mask = new_costs < costs
        positions[better_mask] = new_positions[better_mask]
        costs[better_mask] = new_costs[better_mask]

        # Mise à jour de la meilleure solution globale
        new_global_best_index = np.argmin(costs)
        if costs[new_global_best_index] < global_best_cost:
            global_best_cost = costs[new_global_best_index]
            global_best_position = positions[new_global_best_index].copy()

        # Remplacement aléatoire de nids (solution) avec une probabilité pa
        random_indices = np.random.rand(N) < pa
        positions[random_indices] = np.random.randint(2, size=(np.sum(random_indices), D))
        costs = np.array([func(pos) for pos in positions])

        # Sauvegarder les résultats tous les 'step' itérations
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)

        iteration += 1

    return results