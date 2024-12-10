from decimal import Decimal
import numpy as np

def simulated_annealing_binary1(func, N, D, Tmax, step, initial_temp=Decimal("1000"), alpha=Decimal("0.9")):
    """
    Simulated Annealing for the binary knapsack problem using Decimal.
    """
    # Random initialization of the solution
    current_solution = np.random.randint(2, size=D)
    current_cost = Decimal(func(current_solution))

    # Initial temperature
    temperature = initial_temp
    best_solution = current_solution.copy()
    best_cost = current_cost

    # List to store results at each step
    results = []

    # Main loop of simulated annealing
    for iteration in range(Tmax):
        # Generate a new solution (binary neighborhood)
        new_solution = current_solution.copy()
        flip_index = np.random.randint(D)
        new_solution[flip_index] = 1 - new_solution[flip_index]  # Flip a random bit

        new_cost = Decimal(func(new_solution))

        # Calculate cost difference
        delta_cost = new_cost - current_cost

        # Acceptance criterion
        if delta_cost < 0 or Decimal(np.random.rand()) < (-delta_cost / temperature).exp():
            current_solution = new_solution
            current_cost = new_cost

            # Update the best solution
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

        # Reduce the temperature
        temperature *= alpha

        # Record the best cost at each step
        if (iteration + 1) % step == 0:
            results.append(-best_cost)  # Record the negative best cost for minimization problems

    return results
