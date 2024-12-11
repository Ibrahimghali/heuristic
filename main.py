import matplotlib
from heuristic.utils.select_problem import select_problem
from heuristic.algorithms.genetic_algorithm_with_local_search import genetic_algorithm_with_local_search
from heuristic.algorithms.particle_swarm_optimization_binary_with_local_search import particle_swarm_optimization_binary_with_local_search
from heuristic.algorithms.genetic_algorithm import genetic_algorithm
from heuristic.algorithms.particle_swarm_optimization_binary import particle_swarm_optimization_binary
from heuristic.algorithms.hybrid_ga_pso_with_local_search import hybrid_ga_pso_with_local_search
from heuristic.algorithms.enhanced_hybrid_ga_pso import improved_hybrid_ga_pso
from heuristic.algorithms.cuckoo_search_binary import cuckoo_search_binary
from heuristic.algorithms.harmony_search import harmony_search

import matplotlib.pyplot as plt

import os
import numpy as np 


matplotlib.use('TkAgg')  # Use an interactive backend like 'TkAgg'


def run_algorithm(algorithm, func, N, D, Tmax, step, test_runs):
    """Runs a specified algorithm for a number of test runs."""
    results = [algorithm(func, N, D, Tmax, step) for _ in range(test_runs)]
    best_solutions = [max(run) for run in results]
    mean = np.mean(best_solutions)
    std = np.std(best_solutions)
    return results, best_solutions, mean, std


def save_results(results, algorithm_name, results_dir):
    """Saves the results of an algorithm to a text file."""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"{algorithm_name}_results.txt")
    with open(filepath, "w") as f:
        for run in results:
            f.write(" ".join(map(str, run)) + "\n")


def plot_results(results_dict, Tmax, step, output_dir):
    """Plots the mean results of different algorithms."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for algo_name, results in results_dict.items():
        mean_results = np.mean(results, axis=0)
        plt.plot(range(step, Tmax + 1, step), mean_results, label=algo_name, marker='o')

    plt.xlabel("Iterations")
    plt.ylabel("Objective Function Value")
    plt.title("Algorithm Comparison")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    # Problem selection
    func, D = select_problem()
    if func is None or D is None:
        print("No problem selected. Exiting.")
        exit()

    # Global parameters
    Tmax = 1000
    step = 20
    N = 30
    test_runs = 30
    results_dir = "./results"
    plots_dir = os.path.join(results_dir, "plots")

    # Algorithm execution
    algorithms = {
        #"HGP_LS": hybrid_ga_pso_with_local_search,
        "IHGP": improved_hybrid_ga_pso,
        "CUCKOO": cuckoo_search_binary,
        "PSO": particle_swarm_optimization_binary,
        "PSO_LS": particle_swarm_optimization_binary_with_local_search,
        "GA": genetic_algorithm,
        "GA_LS": genetic_algorithm_with_local_search,
        "HS" : harmony_search
       
    }

    results_dict = {}

    for name, algo in algorithms.items():
        print(f"\nRunning {name}...")
        results, best, mean, std = run_algorithm(algo, func, N, D, Tmax, step, test_runs)
        results_dict[name] = results
        save_results(results, name, results_dir)
        print(f"{name} - Best: {max(best):.5f}, Mean: {mean:.5f}, Std: {std:.5f}")

    # Plot results
    plot_results(results_dict, Tmax, step, plots_dir)
