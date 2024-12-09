import numpy as np
import matplotlib.pyplot as plt
import time


# Algorithme génétique avec recherche locale intégrée
def genetic_algorithm_with_local_search(func, N, D, Tmax, step):
    parents = np.random.randint(0, 2, (N, D))
    fitnesses = np.array([func(parent) for parent in parents])
    results = []

    t = 0
    while t < Tmax:
        mutation_rate = 0.1 - (0.1 - 0.01) * (t / Tmax)
        j, k = np.random.randint(0, N, (2, N))
        cross_point = np.random.randint(1, D - 1)
        enfants = np.hstack((parents[j, :cross_point], parents[k, cross_point:]))

        for i in range(N):
            mutation_mask = np.random.rand(D) < mutation_rate
            enfants[i] = np.where(mutation_mask, 1 - enfants[i], enfants[i])

        enfants_fitnesses = np.array([func(enfant) for enfant in enfants])

        # Recherche locale sur 50 % des meilleurs individus
        num_local_search = max(1, int(0.5 * N))  # 50 % des individus
        #les enfants selon leur fitness et on sélectionne les meilleurs enfants
        best_indices = np.argsort(enfants_fitnesses)[:num_local_search]
        for idx in best_indices:
            individual = enfants[idx]
            # Recherche locale : inverser un bit aléatoire pour améliorer la solution
            for _ in range(5):  # 5 tentatives de recherche locale
                local_individual = individual.copy()
                #On choisit un bit aléatoire à inverser (mutation locale)
                random_bit = np.random.randint(0, D)
                local_individual[random_bit] = 1 - local_individual[random_bit]  # Inversion du bit
                #Évaluation de la nouvelle solution
                local_cost = func(local_individual)
                #Si la nouvelle solution est meilleure, on remplace l'enfant par la nouvelle solution
                if local_cost < enfants_fitnesses[idx]:  # Si l'amélioration est meilleure
                    enfants[idx] = local_individual
                    enfants_fitnesses[idx] = local_cost

        combined_population = np.vstack((parents, enfants))
        combined_fitnesses = np.hstack((fitnesses, enfants_fitnesses))


        # Mise à jour de la population avec les meilleurs individus
        best_indices = np.argsort(combined_fitnesses)[:N]
        parents = combined_population[best_indices]
        fitnesses = combined_fitnesses[best_indices]

        if (t + 1) % step == 0:
            results.append(-fitnesses[0])

        t += 1

    return results