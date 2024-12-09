import matplotlib
matplotlib.use('TkAgg')  # Use an interactive backend like 'TkAgg'

import numpy as np
import matplotlib.pyplot as plt
import time
from binary_dpso_with_memory import binary_dpso_with_memory
from genetic_algorithm import genetic_algorithm
from genetic_algorithm_with_local_search import genetic_algorithm_with_local_search
from particle_swarm_optimization_binary import particle_swarm_optimization_binary
from particle_swarm_optimization_binary_condition import particle_swarm_optimization_binary_condition
from particle_swarm_optimization_binary_with_local_search import particle_swarm_optimization_binary_with_local_search
from select_problem import select_problem   
from MKP import MKP1, MKP2, MKP3, MKP4, MKP5, MKP6, MKP7, MKP8, MKP9, MKP10

if __name__ == "__main__":
    # Sélection du problème
    func, D = select_problem()
    if func is None or D is None:
        exit()

    # Exemple de sélection
    selection = np.random.randint(0, 2, D)
    print(f"Solution aléatoire initiale: {selection}")
    print(f"Valeur de la fonction objectif (avec pénalisation) : {-func(selection)}")

    # Paramètres globaux
    Tmax = 1000
    step = 10
    N = 30
    test_runs = 30

    # PSO
    pso_results = [particle_swarm_optimization_binary(func, N, D, Tmax, step) for _ in range(test_runs)]
    pso_best = [max(run) for run in pso_results]
    pso_mean = np.mean(pso_best)
    pso_std = np.std(pso_best)
    print(f"\nPSO:\n")
    print(f"Meilleure solution finale parmi les 30 répétitions: {max(pso_best):.5f}")
    print(f"Coût moyen des meilleures solutions finales: {pso_mean:.5f}")
    print(f"Écart type des meilleures solutions finales: {pso_std:.5f}")

    # PSO_condition
    # pso_c_results = [particle_swarm_optimization_binary_condition(func, N, D, Tmax, step) for _ in range(test_runs)]
    # pso_c_best = [max(run) for run in pso_c_results]
    # pso_c_mean = np.mean(pso_c_best)
    # pso_c_std = np.std(pso_c_best)
    # print(f"\nPSO_C:\n")
    # print(f"Meilleure solution finale parmi les 30 répétitions: {max(pso_c_best):.5f}")
    # print(f"Coût moyen des meilleures solutions finales: {pso_c_mean:.5f}")
    # print(f"Écart type des meilleures solutions finales: {pso_c_std:.5f}")

    # PSO with Local Search
    pso_ls_results = [particle_swarm_optimization_binary_with_local_search(func, N, D, Tmax, step) for _ in range(test_runs)]
    pso_ls_best = [max(run) for run in pso_ls_results]
    pso_ls_mean = np.mean(pso_ls_best)
    pso_ls_std = np.std(pso_ls_best)
    print(f"\nPSO_LS:\n")
    print(f"Meilleure solution finale parmi les 30 répétitions: {max(pso_ls_best):.5f}")
    print(f"Coût moyen des meilleures solutions finales: {pso_ls_mean:.5f}")
    print(f"Écart type des meilleures solutions finales: {pso_ls_std:.5f}")

    # GA
    ga_results = [genetic_algorithm(func, N, D, Tmax, step) for _ in range(test_runs)]
    ga_best = [max(run) for run in ga_results]
    ga_mean = np.mean(ga_best)
    ga_std = np.std(ga_best)
    print(f"\nGA:\n")
    print(f"Meilleure solution finale parmi les 30 répétitions: {max(ga_best):.5f}")
    print(f"Coût moyen des meilleures solutions finales: {ga_mean:.5f}")
    print(f"Écart type des meilleures solutions finales: {ga_std:.5f}")

    # GA avec recherche locale
    gas_results = [genetic_algorithm_with_local_search(func, N, D, Tmax, step) for _ in range(test_runs)]
    gas_best = [max(run) for run in gas_results]
    gas_mean = np.mean(gas_best)
    gas_std = np.std(gas_best)
    print(f"\nGA_LS:\n")
    print(f"Meilleure solution finale parmi les 30 répétitions: {max(gas_best):.5f}")
    print(f"Coût moyen des meilleures solutions finales: {gas_mean:.5f}")
    print(f"Écart type des meilleures solutions finales: {gas_std:.5f}")

    # BDPSO-M
    # BDPSO_M_results = [binary_dpso_with_memory(func, N, D, Tmax, step) for _ in range(test_runs)]
    # BDPSO_M_best = [max(run) for run in BDPSO_M_results]
    # BDPSO_M_mean = np.mean(BDPSO_M_best)
    # BDPSO_M_std = np.std(BDPSO_M_best)
    # print(f"\nBDPSO-M:\n")
    # print(f"Meilleure solution finale parmi les 30 répétitions: {max(BDPSO_M_best):.5f}")
    # print(f"Coût moyen des meilleures solutions finales: {BDPSO_M_mean:.5f}")
    # print(f"Écart type des meilleures solutions finales: {BDPSO_M_std:.5f}")

    # Sauvegarde des résultats
    with open("results.txt", "w") as f:
        f.write("PSO Results:\n")
        for run in pso_results:
            f.write(" ".join(map(str, run)) + "\n")
        f.write("\nPSO_C Results:\n")
        # for run in pso_c_results:
        #     f.write(" ".join(map(str, run)) + "\n")
        f.write("\nPSO_LS Results:\n")
        for run in pso_ls_results:
            f.write(" ".join(map(str, run)) + "\n")
        f.write("\nGA Results:\n")
        for run in ga_results:
            f.write(" ".join(map(str, run)) + "\n")
        f.write("\nGA_LS Results:\n")
        for run in gas_results:
            f.write(" ".join(map(str, run)) + "\n")
        f.write("\nBDPSO-M Results:\n")
        # for run in BDPSO_M_results:
        #     f.write(" ".join(map(str, run)) + "\n")

    # Graphique des résultats
    plt.figure(figsize=(10, 6))
    plt.plot(range(step, Tmax + 1, step), np.mean(pso_results, axis=0), label='PSO', marker='o')
    # plt.plot(range(step, Tmax + 1, step), np.mean(pso_c_results, axis=0), label='PSO_C', marker='v')
    plt.plot(range(step, Tmax + 1, step), np.mean(pso_ls_results, axis=0), label='PSO_LS', marker='^')
    plt.plot(range(step, Tmax + 1, step), np.mean(ga_results, axis=0), label='GA', marker='x')
    plt.plot(range(step, Tmax + 1, step), np.mean(gas_results, axis=0), label="GA_LS", marker='s')
    # plt.plot(range(step, Tmax + 1, step), np.mean(BDPSO_M_results, axis=0), label="BDPSO-M", marker='d')

    # Ajouter des labels et titre
    plt.xlabel("Itérations")
    plt.ylabel("Valeur de la fonction objectif")
    plt.title("Comparaison des algorithmes (PSO, PSO_C, PSO_LS, GA, GA_LS, BDPSO-M)")
    plt.legend()
    plt.grid()
    plt.show()