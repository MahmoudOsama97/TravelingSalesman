import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
from TravelingSalesman.branch_and_bound import TSPSolverBnB
from TravelingSalesman.dynamic_programming import TSPSolverHeldKarp
from TravelingSalesman.genetic_algorithm import TSPSolverGA
from TravelingSalesman.utils import total_tour_distance


# Function to generate a random TSP instance
def generate_tsp_instance(num_cities, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    coordinates = np.random.rand(num_cities, 2) * 100  # Scale to 0-100 range
    distance_matrix = np.zeros((num_cities, num_cities))
    
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            distance_matrix[i][j] = distance_matrix[j][i] = np.sqrt(dx**2 + dy**2)
    
    return coordinates, distance_matrix

# Function to measure runtime and memory usage
def measure_performance(algorithm, distance_matrix):
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
    start_time = time.time()
    
    solver = algorithm(distance_matrix)
    best_tour, best_cost = solver.solve()
    
    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    runtime = end_time - start_time
    memory_used = mem_after - mem_before
    return best_tour, best_cost, runtime, memory_used

# Function to visualize the best tour
def plot_tour(coordinates, best_tour, title):
    plt.figure(figsize=(8, 6))
    tour_coords = np.array([coordinates[i] for i in best_tour] + [coordinates[best_tour[0]]])
    
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], marker='o', linestyle='-', color='b')
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='r', zorder=2)
    
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.show()

# Main execution
num_cities = 10
coordinates, distance_matrix = generate_tsp_instance(num_cities, seed=42)

# **✅ Debugging Step: Check for Inf or Negative Values in Distance Matrix**
print("[DEBUG] Distance Matrix:\n", distance_matrix)

if np.any(np.isinf(distance_matrix)):
    print("[ERROR] Distance matrix contains 'inf' values! Fix the generation function.")
    exit()

if np.any(distance_matrix < 0):
    print("[ERROR] Distance matrix contains negative values! Fix the generation function.")
    exit()

# Run all three algorithms
results = {}

print("\nRunning Branch and Bound...")
results['BnB'] = measure_performance(TSPSolverBnB, distance_matrix)
print(f"Branch and Bound - Best Cost: {results['BnB'][1]:.2f}, Runtime: {results['BnB'][2]:.4f} sec, Memory: {results['BnB'][3]:.2f} MB")

print("\nRunning Held-Karp (Dynamic Programming)...")
results['Held-Karp'] = measure_performance(TSPSolverHeldKarp, distance_matrix)
print(f"Held-Karp - Best Cost: {results['Held-Karp'][1]:.2f}, Runtime: {results['Held-Karp'][2]:.4f} sec, Memory: {results['Held-Karp'][3]:.2f} MB")

print("\nRunning Genetic Algorithm...")
results['GA'] = measure_performance(TSPSolverGA, distance_matrix)
print(f"Genetic Algorithm - Best Cost: {results['GA'][1]:.2f}, Runtime: {results['GA'][2]:.4f} sec, Memory: {results['GA'][3]:.2f} MB")

# **✅ Debugging Step: Check if best_tour is None before plotting**
if results['Held-Karp'][0] is None:
    print("[ERROR] Held-Karp failed to find a valid tour! Assigning default.")
    results['Held-Karp'] = (list(range(num_cities)), results['Held-Karp'][1])

if results['GA'][0] is None:
    print("[ERROR] Genetic Algorithm failed to find a valid tour! Assigning default.")
    results['GA'] = (list(range(num_cities)), results['GA'][1])

# Plot the best tours
plot_tour(coordinates, results['BnB'][0], "Branch and Bound - Best Tour")
plot_tour(coordinates, results['Held-Karp'][0], "Held-Karp (DP) - Best Tour")
plot_tour(coordinates, results['GA'][0], "Genetic Algorithm - Best Tour")
