import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Fix module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from branch_and_bound import TSPSolverBnB
from dynamic_programming import TSPSolverHeldKarp
from genetic_algorithm import TSPSolverGA
from utils import total_tour_distance

# --- Algorithm wrappers ---
def generate_cities(n):
    return np.random.rand(n, 2) * 100  # Random 2D points in 100x100 plane

def run_held_karp(cities):
    solver = TSPSolverHeldKarp(cities)
    tour = solver.solve()
    return total_tour_distance(cities, tour)

def run_branch_and_bound(cities):
    solver = TSPSolverBnB(cities)
    tour, _ = solver.solve()  # Extract only the tour
    return total_tour_distance(cities, tour)

def run_genetic_algorithm(cities):
    solver = TSPSolverGA(cities, population_size=100, generations=200)
    tour = solver.solve()
    return total_tour_distance(cities, tour)

# --- Main analysis loop ---
city_sizes = [6, 8, 10, 12, 14]
held_karp_times = []
branch_bound_times = []
genetic_times = []

for size in city_sizes:
    print(f"\n[INFO] Testing with {size} cities...")
    cities = generate_cities(size)

    start = time.perf_counter()
    run_held_karp(cities)
    held_karp_times.append(time.perf_counter() - start)
    print(f"Held-Karp time: {held_karp_times[-1]:.4f} s")

    start = time.perf_counter()
    run_branch_and_bound(cities)
    branch_bound_times.append(time.perf_counter() - start)
    print(f"Branch and Bound time: {branch_bound_times[-1]:.4f} s")

    start = time.perf_counter()
    run_genetic_algorithm(cities)
    genetic_times.append(time.perf_counter() - start)
    print(f"Genetic Algorithm time: {genetic_times[-1]:.4f} s")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.semilogy(city_sizes, held_karp_times, 'o-b', label='Held-Karp (DP)')
plt.semilogy(city_sizes, branch_bound_times, 's-g', label='Branch and Bound')
plt.semilogy(city_sizes, genetic_times, '^-r', label='Genetic Algorithm')

plt.xlabel("Number of Cities")
plt.ylabel("Execution Time (seconds, log scale)")
plt.title("TSP Algorithm Performance Comparison")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.figtext(0.5, -0.05, "Machine: AMD 5800X, 32GB RAM, Python 3.12", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("tsp_complexity_comparison.png")
plt.show()
