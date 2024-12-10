import numpy as np
import matplotlib.pyplot as plt
import time


# Algorithme génétique
def genetic_algorithm(func, N, D, Tmax, step):
    parents = np.random.randint(0, 2, (N, D))
    #évalue la performance (fitness) de chaque individu en appliquant la fonction objectif func.
    fitnesses = np.array([func(parent) for parent in parents])
    results = []

    t = 0
    while t < Tmax:
        mutation_rate = 0.1 - (0.1 - 0.01) * (t / Tmax)
        #Choisir deux parents au hasard
        j, k = np.random.randint(0, N, (2, N))
        # couper les deux parents à cet endroit et échanger leurs parties
        cross_point = np.random.randint(1, D - 1)
        #échange de 2 parties pour créer un enfant. np.hstack permet de concaténer les parties des parents pour former un enfant
        enfants = np.hstack((parents[j, :cross_point], parents[k, cross_point:]))

        for i in range(N):
            #compare chaque valeur avec la mutation_rate pour savoir si un élément de l'enfant doit être muté
            mutation_mask = np.random.rand(D) < mutation_rate
            #si mutation_mask est true,on inverse la valeur de l'élément
            enfants[i] = np.where(mutation_mask, 1 - enfants[i], enfants[i])

        enfants_fitnesses = np.array([func(enfant) for enfant in enfants])
        combined_population = np.vstack((parents, enfants))
        combined_fitnesses = np.hstack((fitnesses, enfants_fitnesses))

        #trie les solutions par fitness, et on garde les N meilleures
        best_indices = np.argsort(combined_fitnesses)[:N]
        parents = combined_population[best_indices]
        fitnesses = combined_fitnesses[best_indices]

        if (t + 1) % step == 0:
            results.append(-fitnesses[0])

        t += 1

    return results


