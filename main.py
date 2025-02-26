import argparse
from utils import read_cities_from_csv, create_distance_matrix, total_tour_distance, plot_tour
from branch_and_bound import TSPSolverBnB
from genetic_algorithm import TSPSolverGA
from dynamic_programming import TSPSolverHeldKarp


def main():
    parser = argparse.ArgumentParser(description="TSP Solver")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file containing city data")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="bnb",
        choices=["bnb", "ga", "dp"],
        help="Which algorithm to use: bnb, ga, or dp",
    )
    parser.add_argument("--plot", action="store_true", help="If set, plot the resulting tour using matplotlib")
    args = parser.parse_args()

    # 1. Read the data
    cities = read_cities_from_csv(args.file)
    if len(cities) < 2:
        print("Not enough cities in file.")
        return

    dist_matrix = create_distance_matrix(cities)

    if args.algorithm == "bnb":
        solver = TSPSolverBnB(dist_matrix)
        best_tour, best_cost = solver.solve()
    elif args.algorithm == "dp":
        solver = TSPSolverHeldKarp(dist_matrix)
        best_tour, best_cost = solver.solve()
    elif args.algorithm == "ga":  # Default to GA
        solver = TSPSolverGA(dist_matrix, population_size=50, generations=200, mutation_rate=0.02)
        best_tour, best_cost = solver.solve()
    else:
        print("Invalid algorithm choice.")
        return
    # 2. Print results
    print(f"Best tour found: {best_tour}")
    print(f"Best cost (distance): {best_cost:.4f}")

    # 3. Optionally plot
    if args.plot and best_tour is not None:
        plot_tour(cities, best_tour, title=f"TSP - {args.algorithm.upper()}")


if __name__ == "__main__":
    main()
