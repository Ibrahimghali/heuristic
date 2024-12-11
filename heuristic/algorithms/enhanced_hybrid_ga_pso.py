import numpy as np  # Importation de la bibliothèque NumPy pour les calculs numériques.

def sigmoid(x):
    """Compute the sigmoid function."""
    # Fonction sigmoïde utilisée pour transformer les valeurs continues en probabilités entre 0 et 1.
    return 1 / (1 + np.exp(-x))  # Formule mathématique de la fonction sigmoïde.

def improved_hybrid_ga_pso(func, N, D, Tmax, step):
    """
    Implémentation d'un algorithme hybride amélioré combinant GA (Genetic Algorithm)
    et PSO (Particle Swarm Optimization).
    Improved Hybrid Genetic Algorithm and Particle Swarm Optimization.

    Parameters:
    func (callable): The objective function to minimize.
    N (int): The population size.
    D (int): The dimensionality of the problem.
    Tmax (int): The maximum number of iterations.
    step (int): The step interval for recording results.

    Returns:
    list: The list of best fitness values recorded at each step interval.
    """
    # Initialisation de la population
    positions = np.random.randint(2, size=(N, D))  # Génération aléatoire des positions binaires pour chaque individu.
    velocities = np.random.uniform(-4, 4, (N, D))  # Génération aléatoire des vitesses continues pour chaque individu.
    fitnesses = np.array([func(pos) for pos in positions])  # Évaluation des fitnesses initiales de la population.

    # Initialisation des paramètres pour l'algorithme génétique (GA)
    mutation_rate = 0.2  # Taux de mutation initial.
    crossover_rate = 0.8  # Taux de croisement (crossover).

    # Initialisation des paramètres pour l'optimisation par essaim particulaire (PSO)
    personal_best_positions = positions.copy()  # Meilleures positions individuelles.
    personal_best_costs = fitnesses.copy()  # Meilleurs coûts individuels.
    global_best_index = np.argmin(personal_best_costs)  # Indice du meilleur coût global.
    global_best_position = personal_best_positions[global_best_index].copy()  # Meilleure position globale.
    global_best_cost = personal_best_costs[global_best_index]  # Meilleur coût global.

    results = []  # Liste pour enregistrer les meilleurs résultats à intervalles réguliers.
    iteration = 0  # Compteur d'itérations.

    while iteration < Tmax:  # Boucle principale jusqu'au nombre maximum d'itérations.
        if iteration % 2 == 0:  # Alterne entre PSO et GA (PSO sur les itérations paires).
            # Mise à jour des paramètres PSO
            inertia_weight = 0.9 - 0.4 * (iteration / Tmax)  # Poids d'inertie décroissant.
            cognitive_weight = 2.0  # Coefficient cognitif (influence du meilleur individuel).
            social_weight = 2.0  # Coefficient social (influence du meilleur global).
            r1 = np.random.uniform(0, 1, (N, D))  # Valeurs aléatoires pour le terme cognitif.
            r2 = np.random.uniform(0, 1, (N, D))  # Valeurs aléatoires pour le terme social.

            # Calcul des nouvelles vitesses et positions
            cognitive = cognitive_weight * r1 * (personal_best_positions - positions)
            social = social_weight * r2 * (global_best_position - positions)
            velocities = inertia_weight * velocities + cognitive + social  # Mise à jour des vitesses.
            velocities = np.clip(velocities, -4, 4)  # Limite des vitesses pour éviter des valeurs extrêmes.
            probabilities = sigmoid(velocities)  # Transformation des vitesses en probabilités.
            positions = (probabilities > np.random.uniform(0, 1, (N, D))).astype(int)  # Mise à jour des positions.

            # Mise à jour des fitnesses après le déplacement des particules.
            fitnesses = np.array([func(pos) for pos in positions])
            better_mask = fitnesses < personal_best_costs  # Masque pour les positions améliorées.
            personal_best_costs[better_mask] = fitnesses[better_mask]  # Mise à jour des coûts individuels.
            personal_best_positions[better_mask] = positions[better_mask].copy()  # Mise à jour des positions individuelles.

            # Mise à jour du meilleur global.
            global_best_index = np.argmin(personal_best_costs)
            global_best_cost = personal_best_costs[global_best_index]
            global_best_position = personal_best_positions[global_best_index].copy()
        else:
            # Étape GA (Genetic Algorithm)
            for i in range(N):  # Parcours de chaque individu.
                if np.random.rand() < crossover_rate:  # Probabilité de croisement.
                    parent1, parent2 = np.random.choice(N, 2, replace=False)  # Sélection de deux parents aléatoires.
                    cross_point = np.random.randint(1, D - 1)  # Point de croisement.
                    child = np.hstack((positions[parent1, :cross_point], positions[parent2, cross_point:]))  # Enfant résultant du croisement.
                else:
                    child = positions[i].copy()  # Pas de croisement, clonage de l'individu.

                # Mutation
                mutation_mask = np.random.rand(D) < mutation_rate  # Génération du masque de mutation.
                child = np.where(mutation_mask, 1 - child, child)  # Mutation des gènes sélectionnés.

                # Évaluation du fitness de l'enfant et mise à jour si meilleure solution.
                child_fitness = func(child)
                if child_fitness < fitnesses[i]:
                    positions[i] = child
                    fitnesses[i] = child_fitness

            mutation_rate *= 0.98  # Réduction dynamique du taux de mutation.

        # Mise à jour du meilleur global après GA.
        current_best_index = np.argmin(fitnesses)
        if fitnesses[current_best_index] < global_best_cost:
            global_best_cost = fitnesses[current_best_index]
            global_best_position = positions[current_best_index].copy()

        # Enregistrement des résultats à intervalles réguliers.
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)

        iteration += 1  # Incrémentation du compteur d'itérations.

    return results  # Retourne les résultats enregistrés.

