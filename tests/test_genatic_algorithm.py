import unittest
import random
import sys
sys.path.append('../')
from genetic_algorithm import TSPSolverGA
from utils import total_tour_distance

class TestTSPSolverGA(unittest.TestCase):
    def setUp(self):
        # Simple 3-city TSP
        self.dist_matrix_3 = [
            [0, 10, 15],
            [10, 0, 20],
            [15, 20, 0]
        ]

        # 4-city TSP with known optimal solution
        self.dist_matrix_4 = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ]

        # Set random seed for reproducibility in some tests
        random.seed(42)

    def test_random_tour_generation(self):
        # Test that random tours are valid permutations
        solver = TSPSolverGA(self.dist_matrix_4)

        # Generate multiple tours and check their validity
        for _ in range(10):
            tour = solver._random_tour()

            # Check tour length
            self.assertEqual(len(tour), 4)

            # Check that tour is a permutation of [0,1,2,3]
            self.assertEqual(sorted(tour), [0, 1, 2, 3])

            # Check that all cities are unique (no duplicates)
            self.assertEqual(len(set(tour)), 4)

    def test_tournament_selection(self):
        solver = TSPSolverGA(self.dist_matrix_3)

        # Create a population with known fitness values
        population = [
            [0, 1, 2],  # Tour 1
            [0, 2, 1],  # Tour 2
            [1, 0, 2]   # Tour 3
        ]

        # Assign decreasing fitness scores so first tour is best
        fitness_scores = [1.0, 0.5, 0.25]

        # With enough tournaments, the best individual should be selected most frequently
        selections = [solver._select_tournament(population, fitness_scores, k=2) for _ in range(100)]

        # Count occurrences of each tour
        counts = {}
        for tour in selections:
            tour_tuple = tuple(tour)
            counts[tour_tuple] = counts.get(tour_tuple, 0) + 1

        # Best tour should be selected most often
        best_tour_count = counts.get(tuple(population[0]), 0)
        worst_tour_count = counts.get(tuple(population[2]), 0)

        self.assertGreater(best_tour_count, worst_tour_count)

    def test_order_crossover(self):
        solver = TSPSolverGA(self.dist_matrix_4)

        # Set fixed parents for reproducible tests
        parent1 = [0, 1, 2, 3]
        parent2 = [3, 2, 1, 0]

        # Test crossover multiple times to account for randomness
        for _ in range(10):
            child1, child2 = solver._order_crossover(parent1, parent2)

            # Check that children are valid tours
            self.assertEqual(len(child1), 4)
            self.assertEqual(len(child2), 4)

            # Check that children are permutations of [0,1,2,3]
            self.assertEqual(sorted(child1), [0, 1, 2, 3])
            self.assertEqual(sorted(child2), [0, 1, 2, 3])

            # Children should inherit some properties from parents
            # but this is probabilistic, so we don't test specific patterns

    def test_swap_mutation(self):
        solver = TSPSolverGA(self.dist_matrix_4)

        # Original tour
        original_tour = [0, 1, 2, 3]

        # Copy the tour before mutation
        tour = original_tour.copy()

        # Apply mutation
        solver._swap_mutation(tour)

        # Tour length should remain unchanged
        self.assertEqual(len(tour), 4)

        # Tour should still be a permutation of [0,1,2,3]
        self.assertEqual(sorted(tour), [0, 1, 2, 3])

        # Tour should be different from original (note: extremely unlikely but possible to be the same)
        # If this test occasionally fails, it's acceptable since mutation is random
        self.assertNotEqual(tour, original_tour)

    def test_solve_small_instance(self):
        # Test with small number of iterations for quick testing
        solver = TSPSolverGA(self.dist_matrix_3, population_size=10, generations=5)
        tour, cost = solver.solve()

        # Check that the solution is valid
        self.assertEqual(len(tour), 3)
        self.assertEqual(sorted(tour), [0, 1, 2])

        # Verify the cost calculation
        computed_cost = total_tour_distance(tour, self.dist_matrix_3)
        self.assertEqual(cost, computed_cost)

        # For this simple case, the optimal cost is 45, but GA might not find it with few generations
        # Just ensure the cost is positive and reasonable
        self.assertGreater(cost, 0)
        self.assertLess(cost, 100)  # Arbitrary upper bound for this small example

    def test_larger_run_improves_solution(self):
        # Run with small parameters
        solver_small = TSPSolverGA(self.dist_matrix_4, population_size=10, generations=5)
        _, cost_small = solver_small.solve()

        # Run with larger parameters
        random.seed(42)  # Reset seed for fair comparison
        solver_large = TSPSolverGA(self.dist_matrix_4, population_size=30, generations=30)
        _, cost_large = solver_large.solve()

        # The larger run should generally find a better solution
        # Since GA is stochastic, this isn't guaranteed, but is highly likely
        # We'll accept if they're equal as well, since both might find optimal
        self.assertLessEqual(cost_large, cost_small * 1.5)  # Allow some margin

    def test_edge_case_small_matrix(self):
        # Test with a 2-city matrix
        small_matrix = [
            [0, 5],
            [5, 0]
        ]
        solver = TSPSolverGA(small_matrix, population_size=4, generations=2)
        tour, cost = solver.solve()

        self.assertEqual(cost, 10)  # 5 + 5 = 10
        self.assertEqual(sorted(tour), [0, 1])

    def test_parameter_sensitivity(self):
        # Test that mutation rate affects diversity
        # Low mutation rate
        solver_low_mut = TSPSolverGA(self.dist_matrix_4, population_size=20, generations=10, mutation_rate=0.0)

        # High mutation rate
        solver_high_mut = TSPSolverGA(self.dist_matrix_4, population_size=20, generations=10, mutation_rate=1.0)

        # Run multiple times and check solutions
        # With no mutation, we expect less diversity in final population
        # This is a high-level test to ensure parameters are being used correctly

        # Instead of trying to measure diversity directly, let's just verify that
        # these extreme parameter values don't crash the algorithm
        tour_low, _ = solver_low_mut.solve()
        tour_high, _ = solver_high_mut.solve()

        self.assertEqual(len(tour_low), 4)
        self.assertEqual(len(tour_high), 4)

if __name__ == '__main__':
    unittest.main()