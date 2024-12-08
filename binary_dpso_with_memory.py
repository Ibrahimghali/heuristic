import numpy as np
import matplotlib.pyplot as plt
import time



# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_dpso_with_memory(func, N, D, Tmax, step, memory_size=50):
    """
    Binary DPSO avec mémoire.
    - func : fonction objective à minimiser.
    - N : nombre de particules.
    - D : dimension de chaque solution.
    - Tmax : nombre d'itérations maximales.
    - step : fréquence de sauvegarde des résultats.
    - memory_size : taille maximale de la mémoire.
    """
    # Définir les coefficients en fonction de la dimension
    if D == 28:  # Petite dimension
        cognitive_coeff = 3.30
        social_coeff = 3.30
    elif D == 60:  # Dimension moyenne
        cognitive_coeff = 7.46
        social_coeff = 7.46
    else:  # Grande dimension
        cognitive_coeff = 12
        social_coeff = 12

    # Initialisation
    positions = np.random.randint(2, size=(N, D))  # Solutions initiales binaires
    velocities = np.random.uniform(-4, 4, (N, D))  # Vitesses initiales
    personal_best_positions = positions.copy()
    personal_best_costs = np.array([func(pos) for pos in positions])

    global_best_index = np.argmin(personal_best_costs)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_cost = personal_best_costs[global_best_index]

    memory = []  # Mémoire pour stocker les positions explorées
    results = []

    # Fonctions auxiliaires
    def add_to_memory(position):
        """Ajoute une solution dans la mémoire avec gestion de la taille."""
        memory.append(position.copy())
        if len(memory) > memory_size:
            memory.pop(0)

    def is_in_memory(position):
        """Vérifie si une position est déjà dans la mémoire."""
        for mem_pos in memory:
            if np.array_equal(position, mem_pos):  # Comparaison élément par élément
                return True
        return False

    # Boucle principale
    for iteration in range(Tmax):
        # Mise à jour dynamique du poids d'inertie
        inertia_weight = 0.7 - (0.4 * iteration / Tmax)

        # Génération de coefficients aléatoires
        r1, r2 = np.random.uniform(size=(2, N, D))
        cognitive = cognitive_coeff * r1 * (personal_best_positions - positions)
        social = social_coeff * r2 * (global_best_position - positions)
        velocities = inertia_weight * velocities + cognitive + social

        # Transformation des vitesses en probabilités
        probabilities = sigmoid(velocities)
        random_values = np.random.uniform(size=(N, D))
        positions = (probabilities >= random_values).astype(int)

        # Évaluation des nouvelles solutions
        costs = np.array([func(pos) for pos in positions])

        # Mise à jour des meilleures solutions personnelles
        better_mask = costs < personal_best_costs
        personal_best_costs[better_mask] = costs[better_mask]
        personal_best_positions[better_mask] = positions[better_mask].copy()

        # Mise à jour de la meilleure solution globale
        new_global_best_index = np.argmin(personal_best_costs)
        if personal_best_costs[new_global_best_index] < global_best_cost:
            global_best_cost = personal_best_costs[new_global_best_index]
            global_best_position = personal_best_positions[new_global_best_index].copy()

        # Ajout de la meilleure solution globale dans la mémoire
        add_to_memory(global_best_position)

        # Exploration avec évitement des doublons
        for i in range(N):
            if np.random.rand() < 0.2:  # 20% de chance d'exploration
                new_position = np.random.randint(2, size=D)
                while is_in_memory(new_position):  # Vérifie si déjà visité
                    new_position = np.random.randint(2, size=D)
                positions[i] = new_position
                add_to_memory(new_position)  # Ajoute la nouvelle position validée à la mémoire

        # Sauvegarde des résultats périodiquement
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)

    return results