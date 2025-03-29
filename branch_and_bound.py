import math
import numpy as np

class TSPSolverBnB:
    def __init__(self, cities):
        """
        :param cities: np.ndarray of shape (n, 2), city coordinates
        """
        self.cities = cities
        self.n = len(cities)
        self.dist_matrix = self._compute_distance_matrix()
        self.best_tour = None
        self.best_cost = math.inf

    def _compute_distance_matrix(self):
        matrix = np.full((self.n, self.n), math.inf)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    matrix[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return matrix

    def solve(self):
        initial_path = [0]
        visited = {0}
        initial_matrix = np.copy(self.dist_matrix)
        initial_lb, reduced_matrix = self._reduce_matrix(initial_matrix)

        self._branch_and_bound(
            path=initial_path,
            visited=visited,
            lower_bound=initial_lb,
            current_cost=0.0,
            reduced_matrix=reduced_matrix
        )

        return self.best_tour, self.best_cost

    def _branch_and_bound(self, path, visited, lower_bound, current_cost, reduced_matrix):
        if len(path) == self.n:
            final_cost = current_cost + self.dist_matrix[path[-1]][path[0]]
            if final_cost < self.best_cost:
                self.best_cost = final_cost
                self.best_tour = path + [path[0]]
            return

        if lower_bound >= self.best_cost:
            return

        last_city = path[-1]

        for city in range(self.n):
            if city in visited:
                continue

            cost_to_city = self.dist_matrix[last_city][city]
            new_cost = current_cost + cost_to_city

            new_matrix = np.copy(reduced_matrix)
            new_matrix[last_city, :] = math.inf
            new_matrix[:, city] = math.inf
            new_matrix[city][0] = math.inf  # Don't return to start early

            reduction_cost, reduced = self._reduce_matrix(new_matrix)
            new_lb = lower_bound + cost_to_city + reduction_cost

            if new_lb < self.best_cost:
                visited.add(city)
                path.append(city)
                self._branch_and_bound(path, visited, new_lb, new_cost, reduced)
                path.pop()
                visited.remove(city)

    def _reduce_matrix(self, matrix):
        cost = 0
        n = matrix.shape[0]

        # Row reduction
        for i in range(n):
            row = matrix[i]
            min_val = np.min(row[np.isfinite(row)]) if np.any(np.isfinite(row)) else 0
            if min_val > 0:
                matrix[i] -= min_val
                cost += min_val

        # Column reduction
        for j in range(n):
            col = matrix[:, j]
            min_val = np.min(col[np.isfinite(col)]) if np.any(np.isfinite(col)) else 0
            if min_val > 0:
                matrix[:, j] -= min_val
                cost += min_val

        return cost, matrix
