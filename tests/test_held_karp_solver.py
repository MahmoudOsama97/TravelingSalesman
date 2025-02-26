import unittest
import math
import sys
sys.path.append('../')
from dynamic_programming import TSPSolverHeldKarp

class TestTSPSolverHeldKarp(unittest.TestCase):
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

    def test_held_karp_recursive_base_case(self):
        # Test base case of recursion
        solver = TSPSolverHeldKarp(self.dist_matrix_3)
        # Base case: last_node=1, all cities visited (visited_set_mask=0)
        cost = solver._held_karp_recursive(1, 0)
        self.assertEqual(cost, 10)  # Distance from city 1 back to city 0
        self.assertEqual(solver.memo[(1, 0)], 10)

    def test_held_karp_recursive_memoization(self):
        # Test memoization is working
        solver = TSPSolverHeldKarp(self.dist_matrix_3)
        # Calculate for a specific subproblem
        solver.memo[(2, 0)] = 15  # Mock a memo entry
        cost = solver._held_karp_recursive(2, 0)
        self.assertEqual(cost, 15)  # Should return memoized value

    def test_reconstruct_tour_simple(self):
        # Test tour reconstruction with a small example
        solver = TSPSolverHeldKarp(self.dist_matrix_3)

        # Manually populate memo with a simple case
        # Start at node 0, need to visit nodes 1 and 2
        solver.memo[(1, 0)] = 10  # Cost from 1 back to 0
        solver.memo[(2, 0)] = 15  # Cost from 2 back to 0
        solver.memo[(0, 0b10)] = 15 + 15  # Cost from 0 to 2 + cost from 2 to 0
        solver.memo[(0, 0b01)] = 10 + 10  # Cost from 0 to 1 + cost from 1 to 0
        solver.memo[(1, 0b100)] = 20 + 15  # Cost from 1 to 2 + cost from 2 to 0
        solver.memo[(2, 0b010)] = 20 + 10  # Cost from 2 to 1 + cost from 1 to 0

        # Expect tour [0, 1, 2, 0] with 0 as implicit return
        tour = solver._reconstruct_tour(0, 0b110)  # Visit nodes 1 and 2 from node 0
        self.assertEqual(tour, [0, 1, 2])  # Expected optimal tour

    def test_reconstruct_tour_4_cities(self):
        # Test with 4 cities
        solver = TSPSolverHeldKarp(self.dist_matrix_4)

        # Run the entire algorithm to populate memo
        solver.solve()

        # The solve method has already called _reconstruct_tour
        # Just verify that the best_tour is not None and has correct length
        self.assertIsNotNone(solver.best_tour)
        self.assertEqual(len(solver.best_tour), 4)  # Should visit all 4 cities

        # Verify first city is 0 as expected
        self.assertEqual(solver.best_tour[0], 0)

        # Check that the tour is valid (each city appears exactly once)
        self.assertEqual(sorted(solver.best_tour), [0, 1, 2, 3])

    def test_solve_small_instance(self):
        # Test the complete solve method
        solver = TSPSolverHeldKarp(self.dist_matrix_3)
        tour, cost = solver.solve()

        # For this simple case, we know the optimal tour is 0-1-2-0 with cost 45
        self.assertEqual(cost, 45)
        self.assertEqual(len(tour), 3)
        self.assertEqual(tour[0], 0)  # Tour should start at node 0

        # Verify tour validity
        self.assertEqual(sorted(tour), [0, 1, 2])

    def test_symmetry_preservation(self):
        # Test symmetry in the solutions for symmetric distance matrices
        # Create a symmetric matrix
        symmetric_matrix = [
            [0, 10, 15],
            [10, 0, 20],
            [15, 20, 0]
        ]

        solver = TSPSolverHeldKarp(symmetric_matrix)
        tour, cost = solver.solve()

        # Calculate the cost manually to verify
        manual_cost = 0
        for i in range(len(tour)):
            manual_cost += symmetric_matrix[tour[i]][tour[(i+1) % len(tour)]]

        self.assertEqual(cost, manual_cost)

if __name__ == '__main__':
    unittest.main()