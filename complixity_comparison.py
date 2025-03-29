import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import permutations
import math

# Import algorithms from other files
from dynamic_programming import TSPSolverHeldKarp
from branch_and_bound import TSPSolverBnB
from genetic_algorithm import TSPSolverGA


# Function to generate random Euclidean distance matrix
def generate_random_euclidean_distance_matrix(num_cities):
    # Generate random points in a 2D plane
    points = np.random.rand(num_cities, 2) * 100

    # Compute Euclidean distance matrix
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i, j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))

    return distance_matrix


# Held-Karp algorithm wrapper (exact) - O(n^2 * 2^n) time complexity
def held_karp_tsp(distance_matrix):
    solver = TSPSolverHeldKarp(distance_matrix)
    _, cost = solver.solve()
    return cost


# Branch and Bound algorithm wrapper (exact) - O(n!) worst case but better pruning
def branch_and_bound_tsp(distance_matrix):
    solver = TSPSolverBnB(distance_matrix)
    _, cost = solver.solve()
    return cost


# Genetic Algorithm wrapper (heuristic)
def genetic_algorithm_tsp(distance_matrix):
    solver = TSPSolverGA(distance_matrix)
    _, cost = solver.solve()
    return cost


# Run the experiment and measure execution time
def run_experiment():
    city_sizes = [5, 10, 15]
    algorithms = [
        ("Held-Karp", held_karp_tsp),
        ("Branch and Bound", branch_and_bound_tsp),
        ("Genetic Algorithm", genetic_algorithm_tsp),
    ]

    results = {name: [] for name, _ in algorithms}

    for num_cities in city_sizes:
        print(f"Running experiments for {num_cities} cities...")
        distance_matrix = generate_random_euclidean_distance_matrix(num_cities)

        for name, algorithm in algorithms:
            # Skip brute force for large problems as it would take too long
            if name == "Branch and Bound" and num_cities > 12:
                results[name].append(None)
                print(f"  Skipping {name} for {num_cities} cities (would take too long)")
                continue

            start_time = time.time()
            _ = algorithm(distance_matrix)
            execution_time = time.time() - start_time

            results[name].append(execution_time)
            print(f"  {name}: {execution_time:.6f} seconds")

    return city_sizes, results


# Plot the results
def plot_results(city_sizes, results):
    plt.figure(figsize=(10, 6))
    markers = ["o", "s", "^", "D"]
    colors = ["b", "g", "r", "c"]

    for i, (algorithm, times) in enumerate(results.items()):
        # Filter out None values (skipped runs)
        valid_points = [(size, time) for size, time in zip(city_sizes, times) if time is not None]
        if valid_points:
            valid_sizes, valid_times = zip(*valid_points)
            plt.plot(valid_sizes, valid_times, marker=markers[i], color=colors[i], label=algorithm)

    plt.xlabel("Number of Cities")
    plt.ylabel("Execution Time (seconds)")
    plt.title("TSP Algorithm Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")  # Use logarithmic scale for better visualization

    # Add machine specifications
    plt.figtext(
        0.5,
        0,
        "Machine specifications: AMD 5800X CPU, 32GB RAM, Python 3.11",
        ha="center",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("tsp_complexity_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    print("TSP Algorithm Performance Comparison")
    print("------------------------------------")
    print("Machine specifications: AMD 5800X CPU, 32GB RAM, Python 3.11")

    city_sizes, results = run_experiment()
    plot_results(city_sizes, results)
