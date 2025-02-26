import math
import sys


class TSPSolverBnB:
    def __init__(self, dist_matrix):
        self.dist_matrix = dist_matrix
        self.n = len(dist_matrix)
        self.best_tour = None
        self.best_cost = math.inf

    def solve(self):
        """
        Public method to solve TSP using Branch-and-Bound and return the best tour and cost.
        """
        # Start with a trivial node 0 as a root
        initial_path = [0]
        visited = {0}  # Use set for efficient lookups
        current_cost = 0.0

        # Precompute a lower-bound from the initial reduced matrix
        initial_lb, initial_reduced_matrix = self._reduce_matrix(
            [row[:] for row in self.dist_matrix]
        )  # copy then reduce

        self._branch_and_bound(
            path=initial_path,
            visited=visited,
            lower_bound=initial_lb,
            current_cost=current_cost,
            reduced_matrix=initial_reduced_matrix,  # Use the reduced matrix from initial reduction
        )

        return self.best_tour, self.best_cost

    def _branch_and_bound(self, path, visited, lower_bound, current_cost, reduced_matrix):
        """
        Recursive function to explore possible paths with bounding.
        """
        # If all cities are visited, close the tour to the start city
        if len(path) == self.n:
            tour_cost = current_cost + self.dist_matrix[path[-1]][path[0]]
            if tour_cost < self.best_cost:
                self.best_cost = tour_cost
                self.best_tour = path[:]
            return

        # If the best bound is already worse than an existing solution, prune
        if lower_bound >= self.best_cost:
            return

        # Explore next cities
        last_city_in_path = path[-1] if path else None  # Handle initial case

        for city in range(self.n):
            if city not in visited:
                cost_to_city = (
                    self.dist_matrix[last_city_in_path][city] if last_city_in_path is not None else 0
                )  # Cost from last city to current
                new_cost = current_cost + cost_to_city

                # Build a new reduced matrix for the next step
                new_matrix = [row[:] for row in reduced_matrix]

                # Block returning to the immediately previous city to prevent cycles of length 2
                if last_city_in_path is not None:
                    new_matrix[city][last_city_in_path] = math.inf

                # Reduce the new matrix and get reduction cost
                red_cost, new_matrix = self._reduce_matrix(new_matrix)
                new_lb = lower_bound + red_cost  # Update lower bound using reduction cost

                # Recurse
                visited.add(city)
                path.append(city)

                self._branch_and_bound(path, visited, new_lb, new_cost, new_matrix)

                visited.remove(city)
                path.pop()

    def _reduce_matrix(self, matrix):
        """
        Reduces each row and column to have at least one zero if possible,
        and returns the sum of reductions (lower bound contribution) plus the reduced matrix.
        """
        n = len(matrix)
        row_reduction = [0] * n
        col_reduction = [0] * n

        # Row reduction
        for i in range(n):
            row_min = min(matrix[i])
            if row_min > 0 and row_min != math.inf:
                for j in range(n):
                    if matrix[i][j] != math.inf:
                        matrix[i][j] -= row_min
                row_reduction[i] = row_min

        # Column reduction
        for j in range(n):
            col_min = math.inf
            for i in range(n):
                if matrix[i][j] < col_min:
                    col_min = matrix[i][j]
            if col_min > 0 and col_min != math.inf:
                for i in range(n):
                    if matrix[i][j] != math.inf:
                        matrix[i][j] -= col_min
                col_reduction[j] = col_min

        total_reduction = sum(row_reduction) + sum(col_reduction)
        return total_reduction, matrix
