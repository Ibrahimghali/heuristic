from decimal import Decimal  # Importation de la classe Decimal pour une précision accrue si nécessaire
import numpy as np  # Importation de NumPy pour les calculs numériques rapides

def cuckoo_search_binary(func, N, D, Tmax, step, pa=0.25, alpha=0.01):
    """
    Perform binary Cuckoo Search optimization.

    Parameters:
    func (callable): The objective function to be minimized.
    N (int): Number of nests (solutions).
    D (int): Dimensionality of the problem.
    Tmax (int): Maximum number of iterations.
    step (int): Interval for recording results.
    pa (float, optional): Probability of abandoning a nest. Default is 0.25.
    alpha (float, optional): Step size scaling factor. Default is 0.01.

    Returns:
    list: A list of the best objective function values found at each 'step' interval.
    """
    # Initialiser les positions des nids (solutions binaires aléatoires)
    positions = np.random.randint(2, size=(N, D))

    # Calculer les coûts associés à chaque solution initiale
    costs = np.array([func(pos) for pos in positions])

    # Déterminer la meilleure solution initiale
    global_best_index = np.argmin(costs)  # Indice de la solution avec le coût minimal
    global_best_position = positions[global_best_index].copy()  # Position de la meilleure solution
    global_best_cost = costs[global_best_index]  # Coût de la meilleure solution

    # Initialisation de la liste pour stocker les résultats
    results = []

    # Boucle principale pour Tmax itérations
    iteration = 0
    while iteration < Tmax:
        # Générer de nouvelles positions pour chaque nid
        new_positions = positions.copy()  # Copier les positions actuelles
        for i in range(N):  # Pour chaque nid
            step_size = alpha * np.random.randn(D)  # Générer un pas aléatoire
            new_positions[i] = (positions[i] + step_size) % 2  # Modifier la position de manière binaire

        # Calculer les coûts des nouvelles positions
        new_costs = np.array([func(pos) for pos in new_positions])

        # Remplacer les anciennes solutions par les nouvelles si elles sont meilleures
        better_mask = new_costs < costs  # Identifier les solutions améliorées
        positions[better_mask] = new_positions[better_mask]  # Mettre à jour les positions
        costs[better_mask] = new_costs[better_mask]  # Mettre à jour les coûts

        # Mise à jour de la meilleure solution globale
        new_global_best_index = np.argmin(costs)  # Trouver la meilleure solution actuelle
        if costs[new_global_best_index] < global_best_cost:  # Si elle est meilleure que l'ancienne
            global_best_cost = costs[new_global_best_index]  # Mettre à jour le coût global
            global_best_position = positions[new_global_best_index].copy()  # Mettre à jour la position globale

        # Abandonner certains nids aléatoirement avec une probabilité `pa`
        random_indices = np.random.rand(N) < pa  # Sélectionner des nids aléatoires à remplacer
        positions[random_indices] = np.random.randint(2, size=(np.sum(random_indices), D))  # Réinitialiser ces nids
        costs = np.array([func(pos) for pos in positions])  # Recalculer leurs coûts

        # Enregistrer le coût de la meilleure solution toutes les `step` itérations
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)  # Ajouter le coût global inversé (pour maximisation)

        iteration += 1  # Passer à l'itération suivante

    return results  # Retourner les résultats enregistrés
