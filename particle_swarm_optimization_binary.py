import numpy as np
import matplotlib.pyplot as plt
import time


# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def particle_swarm_optimization_binary(func, N, D,Tmax, step):
    positions = np.random.randint(2, size=(N, D))
    velocities = np.random.uniform(-6, 6, (N, D))
    #copie des positions initiales des particules
    personal_best_positions = positions.copy()
    #calcule le coût (fonction objectif) pour chaque particule à partir de ses positions initiales.
    personal_best_costs = np.array([func(pos) for pos in positions])

    global_best_index = np.argmin(personal_best_costs)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_cost = personal_best_costs[global_best_index]




    iteration = 0
    results = []

    while iteration < Tmax:
        inertia_weight = 0.9 - 0.5 * iteration / Tmax   #Ce poids contrôle combien la vitesse précédente influence le mouvement actuel de la particule.Il diminue progressivement au fil des itérations.

        r1 = np.random.uniform(0, 1, (N, D))
        # r1 =  2
        # r2 = 2
        r2 = np.random.uniform(0, 1, (N, D))

        cognitive = 1.5 * r1 * (personal_best_positions - positions)   #Encourage chaque particule à se rapprocher de sa meilleure solution personnelle.
        social = 1.5 * r2 * (global_best_position - positions)
        velocities = inertia_weight * velocities + cognitive + social    #Encourage chaque particule à se rapprocher de la meilleure solution globale.

        probabilities = sigmoid(velocities)    #Les vitesses sont converties en probabilités à l’aide de la fonction Sigmoid.
        random_values = np.random.uniform(0, 1, (N, D))
        positions = (probabilities >= random_values).astype(int)    #Si la probabilité est supérieure ou égale à la valeur aléatoire, la position devient 1.Sinon, elle reste 0.

        costs = np.array([func(pos) for pos in positions])

        better_mask = costs < personal_best_costs     #identifie les particules ayant trouvé une solution meilleure que leur solution personnelle précédente.
        personal_best_costs[better_mask] = costs[better_mask]
        personal_best_positions[better_mask] = positions[better_mask].copy()

        new_global_best_index = np.argmin(personal_best_costs)
        if personal_best_costs[new_global_best_index] < global_best_cost:     #Si une particule trouve une solution meilleure que la solution globale actuelle, cette dernière est mise à jour.
            global_best_cost = personal_best_costs[new_global_best_index]
            global_best_position = personal_best_positions[new_global_best_index].copy()

        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)

        iteration += 1

    return results