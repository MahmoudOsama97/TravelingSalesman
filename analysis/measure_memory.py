import tracemalloc

def measure_memory_usage(algorithm, *args):
    tracemalloc.start()  # Start tracking memory
    result = algorithm(*args)  # Run the algorithm
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop tracking memory
    return result, peak / 1024 / 1024  # Convert to MB