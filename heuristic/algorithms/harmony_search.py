from decimal import Decimal
import numpy as np
import time




def harmony_search(func, N, D, Tmax, step, HMS=10, HMCR=0.9, PAR=0.3, BW=0.01):
    """
    Algorithme de Recherche Harmonique (Harmony Search).
    :param func: Fonction objectif à minimiser. Elle prend une solution (vecteur) comme argument.
    :param N: Nombre d'objets (non utilisé directement ici, mais correspond au problème défini).
    :param D: Taille d'une solution (dimension de recherche).
    :param Tmax: Nombre maximal d'itérations (critère d'arrêt principal).
    :param step: Intervalle pour enregistrer les résultats intermédiaires.
    :param HMS: Taille de la mémoire harmonique (nombre de solutions conservées simultanément).
    :param HMCR: Taux de considération de la mémoire harmonique (probabilité d'utiliser une solution existante).
    :param PAR: Taux d'ajustement de pas (Pitch Adjustment Rate, probabilité d'ajuster les solutions sélectionnées).
    :param BW: Bande de réglage (Bandwidth) utilisée pour l'ajustement de pas.
    :return:
        - results: Liste des meilleures valeurs de la fonction objectif à chaque intervalle `step`.
        - global_best_solution: Meilleure solution trouvée pendant l'algorithme.
        - global_best_cost: Meilleur coût associé à la meilleure solution.
    """
    # Initialiser la mémoire harmonique avec HMS solutions générées aléatoirement
    harmony_memory = np.random.randint(0, 2, size=(HMS, D))  # Matrice HMS x D (binaire : 0 ou 1)

    # Calculer les coûts de chaque solution initiale
    costs = np.array([func(harmony) for harmony in harmony_memory])  # Évaluer toutes les solutions initiales

    # Identifier la meilleure solution initiale dans la mémoire harmonique
    best_index = np.argmin(costs)  # Trouver l'indice de la solution ayant le plus petit coût
    global_best_solution = harmony_memory[best_index].copy()  # Copier cette solution comme la meilleure globale
    global_best_cost = costs[best_index]  # Coût associé à la meilleure solution

    results = []  # Liste pour stocker les résultats intermédiaires
    iteration = 0  # Compteur d'itérations

    # Boucle principale : itérations jusqu'à Tmax
    while iteration < Tmax:
        # Générer une nouvelle solution
        new_harmony = []  # Nouvelle solution vide
        for i in range(D):  # Pour chaque dimension de la solution
            if np.random.rand() < HMCR:  # Décider si on utilise la mémoire harmonique
                # Sélectionner un élément dans la mémoire harmonique pour cette dimension
                selected_harmony = harmony_memory[np.random.randint(HMS), i]
                if np.random.rand() < PAR:  # Ajustement de pas (Pitch Adjustment)
                    # Inverser le bit avec une probabilité BW
                    selected_harmony = 1 - selected_harmony if np.random.rand() < BW else selected_harmony
                new_harmony.append(selected_harmony)  # Ajouter l'élément ajusté ou sélectionné
            else:
                # Génération aléatoire d'un élément si on n'utilise pas la mémoire harmonique
                new_harmony.append(np.random.randint(0, 2))

        # Convertir la nouvelle solution en tableau numpy
        new_harmony = np.array(new_harmony)

        # Calculer le coût de la nouvelle solution
        new_cost = func(new_harmony)

        # Mise à jour de la mémoire harmonique
        if new_cost < costs.max():  # Si la nouvelle solution est meilleure que la pire solution actuelle
            worst_index = np.argmax(costs)  # Trouver l'indice de la pire solution
            harmony_memory[worst_index] = new_harmony  # Remplacer la pire solution par la nouvelle
            costs[worst_index] = new_cost  # Mettre à jour le coût correspondant

            # Mise à jour de la meilleure solution globale
            if new_cost < global_best_cost:  # Si la nouvelle solution est meilleure que la meilleure solution globale
                global_best_cost = new_cost
                global_best_solution = new_harmony.copy()

        # Enregistrer les résultats toutes les `step` itérations
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)  # Stocker le coût négatif pour maximisation

        iteration += 1  # Incrémenter le compteur d'itérations

    # Retourner les résultats intermédiaires, la meilleure solution et son coût
    return results
