import time

def measure_runtime(algorithm, *args):
    start_time = time.time()  # Start timer
    result = algorithm(*args)  # Run the algorithm
    end_time = time.time()  # End timer
    runtime = end_time - start_time  # Calculate runtime
    return result, runtime