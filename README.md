# Traveling Salesman Problem Solver

A comprehensive implementation of multiple algorithms to solve the classic Traveling Salesman Problem (TSP), featuring command-line and graphical user interfaces.

![TSP Solver Interface](https://upload.wikimedia.org/wikipedia/commons/2/2b/Bruteforce.gif)

## Overview

The Traveling Salesman Problem is a fundamental combinatorial optimization problem: given a list of cities and the distances between them, find the shortest possible route that visits each city exactly once and returns to the origin city. This repository implements three different approaches to solve TSP:

- **Dynamic Programming (Held-Karp algorithm)** - Optimal solution with O(2^n × n^2) complexity
- **Branch and Bound** - Optimal solution with pruning to improve average-case performance
- **Genetic Algorithm** - Approximate solution with good performance on larger instances

## Installation

```bash
# Clone the repository
git clone https://github.com/MahmoudOsama97/TravelingSalesman.git
# Navigate to the project directory
cd TravelingSalesman
# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The project provides a command-line interface to solve TSP instances:

```bash
python main.py --file path/to/cities.csv --algorithm [bnb|dp|ga] [--plot]
```

Parameters:
- `--file`: Path to CSV file with city data (required)
- `--algorithm`: Algorithm choice - `bnb` (Branch and Bound), `dp` (Dynamic Programming), or `ga` (Genetic Algorithm)
- `--plot`: Optional flag to display the resulting tour visualization

### Graphical User Interface

For a more interactive experience, run the Gradio interface:

```bash
python demo.py
```

The interface allows you to:
- Generate random city configurations
- Choose between different algorithms
- Visualize the optimal tour with distances
- Reset and try different configurations

![Gradio Interface](https://i.imgur.com/example.png)

## Data Format

Input CSV files should have the following format:

```
city_id,x_coordinate,y_coordinate
City1,10.0,20.0
City2,15.0,25.0
...
```

## Algorithms

### Dynamic Programming (Held-Karp)
- Guarantees optimal solution
- Good performance for small to medium instances (up to ~20-25 cities)
- Uses memoization to avoid redundant calculations
- Time complexity: O(2^n × n^2)
- Space complexity: O(2^n × n)

### Branch and Bound
- Guarantees optimal solution
- Uses lower bounds to prune unpromising branches
- Often performs better than DP on sparse graphs
- Matrix reduction technique for tight lower bounds

### Genetic Algorithm
- Approximate solution, not guaranteed to be optimal
- Significantly faster for large instances
- Configurable parameters (population size, generations, mutation rate)
- Based on natural selection principles with order crossover and swap mutation

## Project Structure

```
TravelingSalesman/
├── branch_and_bound.py
├── dynamic_programming.py
├── genetic_algorithm.py
├── utils.py
├── main.py
├── demo.py
├── tests/
│   ├── test_branch_and_bound_solver.py
│   ├── test_held_karp_solver.py
│   ├── test_genetic_algorithm.py
└── README.md
```

## Testing

Run unit tests to verify algorithm implementations:

```bash
python -m unittest discover -s tests
```

The test suite includes verification of:
- Correctness on small instances with known optimal solutions
- Base cases and recursion logic
- Edge cases such as 2-city problems
- Matrix reduction in Branch and Bound
- Tour reconstruction logic
- Genetic operators (selection, crossover, mutation)

## Performance Comparison

| Algorithm | Small (n=10) | Medium (n=20) | Large (n=50) |
|-----------|--------------|---------------|--------------|
| DP        | < 1 sec      | ~10-20 sec    | Memory limit |
| BnB       | < 1 sec      | ~5-15 sec     | Varies       |
| GA        | < 1 sec      | < 1 sec       | ~10-30 sec   |

## Contributors

This project was created as part of the Advanced Algorithms course at UBC.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

