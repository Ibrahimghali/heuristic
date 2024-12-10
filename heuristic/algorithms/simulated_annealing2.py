from decimal import Decimal
import numpy as np
import time

def simulated_annealing_binary2(func, N, D, Tmax, step, initial_temp=1000, alpha=0.98):
    """
    Simulated Annealing pour le problème de sac à dos binaire avec la fonction MKP1.
    """
    # Initialisation aléatoire de la solution
    current_solution = np.random.randint(2, size=D)
    current_cost = Decimal(func(current_solution))

    # Initialisation de la température
    temperature = Decimal(initial_temp)
    best_solution = current_solution.copy()
    best_cost = current_cost

    # Liste des résultats pour chaque itération
    results = []

    # Boucle principale du recuit simulé
    for iteration in range(Tmax):
        # Générer une nouvelle solution voisine (voisinage binaire)
        new_solution = current_solution.copy()
        flip_index = np.random.randint(D)
        new_solution[flip_index] = 1 - new_solution[flip_index]  # Inverser un bit aléatoire

        new_cost = Decimal(func(new_solution))

        # Calcul de la différence de coût
        delta_cost = new_cost - current_cost

        # Critère d'acceptation
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / temperature):
            current_solution = new_solution.copy()
            current_cost = new_cost

            # Mise à jour de la meilleure solution trouvée
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost

        # Diminution de la température
        temperature *= Decimal(alpha)

        # Enregistrement des résultats à chaque étape définie
        if (iteration + 1) % step == 0:
            results.append(-best_cost)

    return results