import unittest
import math
import sys
sys.path.append('../')
from branch_and_bound import TSPSolverBnB

class TestTSPSolverBnB(unittest.TestCase):
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

    def test_reduce_matrix(self):
        # Test the matrix reduction function
        solver = TSPSolverBnB(self.dist_matrix_3)

        # Create a test matrix that needs reduction
        test_matrix = [
            [math.inf, 10, 15],
            [10, math.inf, 20],
            [15, 20, math.inf]
        ]

        reduction, reduced_matrix = solver._reduce_matrix([row[:] for row in test_matrix])

        # Check that reduction value is correct (sum of row and column minimums)
        self.assertEqual(reduction, 40)  # (10+10+15) + (0+0+5) = 40

        # Check that the reduced matrix has at least one zero in each row and column
        for row in reduced_matrix:
            self.assertIn(0, [val for val in row if val != math.inf])

        # Check column-wise
        for j in range(3):
            col_has_zero = False
            for i in range(3):
                if reduced_matrix[i][j] == 0:
                    col_has_zero = True
                    break
            self.assertTrue(col_has_zero)

    def test_branch_and_bound_small(self):
        # Test the core branch and bound logic with a small example
        solver = TSPSolverBnB(self.dist_matrix_3)

        # Set up initial state for branch and bound
        path = [0]
        visited = {0}
        current_cost = 0
        reduction, reduced_matrix = solver._reduce_matrix([row[:] for row in self.dist_matrix_3])

        # Run branch and bound
        solver._branch_and_bound(path, visited, reduction, current_cost, reduced_matrix)

        # Check results
        self.assertIsNotNone(solver.best_tour)
        self.assertEqual(len(solver.best_tour), 3)
        self.assertEqual(solver.best_tour[0], 0)
        self.assertEqual(solver.best_cost, 45)  # Expected optimal cost for this problem

    def test_solve_small_instance(self):
        # Test the complete solve method
        solver = TSPSolverBnB(self.dist_matrix_3)
        tour, cost = solver.solve()

        # For this simple case, we know the optimal tour is 0-1-2-0 with cost 45
        self.assertEqual(cost, 45)
        self.assertEqual(len(tour), 3)
        self.assertEqual(tour[0], 0)  # Tour should start at node 0

        # Verify tour validity
        self.assertEqual(sorted(tour), [0, 1, 2])

    def test_solve_medium_instance(self):
        # Test with 4 cities
        solver = TSPSolverBnB(self.dist_matrix_4)
        tour, cost = solver.solve()

        # Verify the tour has all cities
        self.assertIsNotNone(tour)
        self.assertEqual(len(tour), 4)
        self.assertEqual(sorted(tour), [0, 1, 2, 3])

        # Calculate the cost manually to verify
        manual_cost = 0
        for i in range(len(tour)):
            manual_cost += self.dist_matrix_4[tour[i]][tour[(i+1) % len(tour)]]

        self.assertEqual(cost, manual_cost)

        # Expected optimal cost for this problem
        self.assertEqual(cost, 80)

    def test_edge_case_small_matrix(self):
        # Test with a 2-city matrix
        small_matrix = [
            [0, 5],
            [5, 0]
        ]
        solver = TSPSolverBnB(small_matrix)
        tour, cost = solver.solve()

        self.assertEqual(cost, 10)  # 5 + 5 = 10
        self.assertEqual(sorted(tour), [0, 1])

    def test_symmetry_preservation(self):
        # Test symmetry in the solutions for symmetric distance matrices
        symmetric_matrix = [
            [0, 10, 15],
            [10, 0, 20],
            [15, 20, 0]
        ]

        solver = TSPSolverBnB(symmetric_matrix)
        tour, cost = solver.solve()

        # Calculate the cost manually to verify
        manual_cost = 0
        for i in range(len(tour)):
            manual_cost += symmetric_matrix[tour[i]][tour[(i+1) % len(tour)]]

        self.assertEqual(cost, manual_cost)

        # The forward and reverse tours should have the same cost
        reverse_tour = tour[::-1]
        reverse_cost = 0
        for i in range(len(reverse_tour)):
            reverse_cost += symmetric_matrix[reverse_tour[i]][reverse_tour[(i+1) % len(reverse_tour)]]

        self.assertEqual(cost, reverse_cost)

if __name__ == '__main__':
    unittest.main()