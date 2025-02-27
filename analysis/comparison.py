import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt
from functools import wraps

# =============================================
# Helper Functions
# =============================================

def generate_tsp_instance(num_cities, seed=None):
    """
    Generate a random symmetric TSP instance with Euclidean distances.
    :param num_cities: Number of cities.
    :param seed: Random seed for reproducibility.
    :return: Distance matrix (numpy array).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random city coordinates in a 2D plane
    coordinates = np.random.rand(num_cities, 2) * 100  # Scale to 0-100 range
    
    # Calculate Euclidean distances
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            distance_matrix[i][j] = distance_matrix[j][i] = np.sqrt(dx**2 + dy**2)
    
    return distance_matrix

def timeout(seconds):
    """
    Decorator to stop a function if it runs for too long.
    :param seconds: Maximum allowed runtime in seconds.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = None
            
            def target():
                nonlocal result
                result = func(*args, **kwargs)
            
            import threading
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                print(f"Function {func.__name__} timed out after {seconds} seconds.")
                return None
            else:
                return result
        return wrapper
    return decorator

def measure_runtime(algorithm, *args):
    """
    Measure the runtime of an algorithm.
    :param algorithm: The algorithm function to measure.
    :param args: Arguments to pass to the algorithm.
    :return: Result of the algorithm and runtime in seconds.
    """
    start_time = time.time()  # Start timer
    result = algorithm(*args)  # Run the algorithm
    end_time = time.time()  # End timer
    runtime = end_time - start_time  # Calculate runtime
    return result, runtime

def measure_memory_usage(algorithm, *args):
    """
    Measure the memory usage of an algorithm.
    :param algorithm: The algorithm function to measure.
    :param args: Arguments to pass to the algorithm.
    :return: Result of the algorithm and peak memory usage in MB.
    """
    tracemalloc.start()  # Start tracking memory
    result = algorithm(*args)  # Run the algorithm
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop tracking memory
    return result, peak / 1024 / 1024  # Convert to MB

# =============================================
# Algorithm Implementations (Replace with your actual implementations)
# =============================================

@timeout(60)  # Set a timeout of 60 seconds
def branch_and_bound(distance_matrix):
    """
    Placeholder for Branch and Bound algorithm.
    Replace this with your actual implementation.
    """
    # Simulate some work
    time.sleep(0.01 * len(distance_matrix)**2)
    return 0  # Return a dummy result

@timeout(60)  # Set a timeout of 60 seconds
def held_karp(distance_matrix):
    """
    Placeholder for Held-Karp algorithm.
    Replace this with your actual implementation.
    """
    # Simulate some work
    time.sleep(0.02 * 2**len(distance_matrix))
    return 0  # Return a dummy result

@timeout(60)  # Set a timeout of 60 seconds
def genetic_algorithm(distance_matrix):
    """
    Placeholder for Genetic Algorithm.
    Replace this with your actual implementation.
    """
    # Simulate some work
    time.sleep(0.05 * len(distance_matrix))
    return 0  # Return a dummy result

# =============================================
# Experiment Functions
# =============================================

def generate_runtime_data(num_cities_list, seed=42):
    """
    Generate runtime data for Branch and Bound, Held-Karp, and Genetic Algorithm.
    :param num_cities_list: List of numbers of cities to test.
    :param seed: Random seed for reproducibility.
    :return: List of dictionaries containing runtime data.
    """
    runtime_data = []
    for num_cities in num_cities_list:
        print(f"Running experiments for {num_cities} cities...")
        
        # Generate test data
        distance_matrix = generate_tsp_instance(num_cities, seed)
        
        # Test Branch and Bound
        print("Running Branch and Bound...")
        result_bnb, runtime_bnb = measure_runtime(branch_and_bound, distance_matrix)
        if result_bnb is None:
            print("Branch and Bound timed out.")
            runtime_bnb = None
        
        # Test Held-Karp
        print("Running Held-Karp...")
        result_hk, runtime_hk = measure_runtime(held_karp, distance_matrix)
        if result_hk is None:
            print("Held-Karp timed out.")
            runtime_hk = None
        
        # Test Genetic Algorithm
        print("Running Genetic Algorithm...")
        result_ga, runtime_ga = measure_runtime(genetic_algorithm, distance_matrix)
        if result_ga is None:
            print("Genetic Algorithm timed out.")
            runtime_ga = None
        
        # Store results
        runtime_data.append({
            "num_cities": num_cities,
            "bnb_runtime": runtime_bnb,
            "hk_runtime": runtime_hk,
            "ga_runtime": runtime_ga,
        })
    
    return runtime_data

def plot_complexity_comparison(runtime_data, filename=None):
    """
    Plot the complexity comparison graph.
    :param runtime_data: List of dictionaries containing runtime data.
    :param filename: If provided, save the graph to this file.
    """
    # Extract data
    num_cities = [data["num_cities"] for data in runtime_data]
    bnb_runtimes = [data["bnb_runtime"] if data["bnb_runtime"] is not None else None for data in runtime_data]
    hk_runtimes = [data["hk_runtime"] if data["hk_runtime"] is not None else None for data in runtime_data]
    ga_runtimes = [data["ga_runtime"] if data["ga_runtime"] is not None else None for data in runtime_data]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(num_cities, bnb_runtimes, label="Branch and Bound", marker="o", linestyle="-", color="blue")
    plt.plot(num_cities, hk_runtimes, label="Held-Karp", marker="o", linestyle="-", color="red")
    plt.plot(num_cities, ga_runtimes, label="Genetic Algorithm", marker="o", linestyle="-", color="green")
    
    # Add labels and title
    plt.xlabel("Number of Cities")
    plt.ylabel("Runtime (seconds)")
    plt.title("Complexity Comparison: Runtime vs. Number of Cities")
    plt.legend()
    plt.grid()
    
    # Save or show the graph
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

# =============================================
# Main Script
# =============================================

if __name__ == "__main__":
    # Define the number of cities to test
    num_cities_list = [5, 10, 15]  # Reduced number of cities for testing
    
    # Generate runtime data
    runtime_data = generate_runtime_data(num_cities_list)
    
    # Print runtime data
    print("\nRuntime Data:")
    for data in runtime_data:
        print(data)
    
    # Plot and save the complexity comparison graph
    plot_complexity_comparison(runtime_data, filename="complexity_comparison.png")
    print("\nComplexity comparison graph saved as 'complexity_comparison.png'.")