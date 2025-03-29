import numpy as np
import sys

class TSPSolverHeldKarp:
    def __init__(self, cities):
        self.cities = cities
        self.n = len(cities)
        self.dist_matrix = self._compute_distance_matrix()
        self.memo = {}
        self.best_cost = float('inf')
        self.best_path = []

    def _compute_distance_matrix(self):
        matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    matrix[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return matrix

    def solve(self):
        # Start from node 0, with only node 0 visited
        start_node = 0
        initial_set_mask = 1 << start_node
        self.best_cost = self._held_karp_recursive(start_node, initial_set_mask)
        path = self._reconstruct_path()
        return path

    def _held_karp_recursive(self, last_node, visited_set_mask):
        if visited_set_mask == (1 << self.n) - 1:
            # All cities visited, return to start
            return self.dist_matrix[last_node][0]

        key = (last_node, visited_set_mask)
        if key in self.memo:
            return self.memo[key]

        min_cost = float('inf')
        for next_node in range(self.n):
            if not visited_set_mask & (1 << next_node):
                next_visited_set_mask = visited_set_mask | (1 << next_node)
                cost = self.dist_matrix[last_node][next_node] + \
                       self._held_karp_recursive(next_node, next_visited_set_mask)
                if cost < min_cost:
                    min_cost = cost

        self.memo[key] = min_cost
        return min_cost

    def _reconstruct_path(self):
        """Optional: Reconstruct path from memo (not used in timing)."""
        visited_set_mask = 1
        current_node = 0
        path = [current_node]

        while visited_set_mask != (1 << self.n) - 1:
            best_next = None
            min_cost = float('inf')

            for next_node in range(self.n):
                if visited_set_mask & (1 << next_node):
                    continue
                next_visited_set_mask = visited_set_mask | (1 << next_node)
                key = (next_node, next_visited_set_mask)
                if (current_node, visited_set_mask) in self.memo and key in self.memo:
                    cost = self.dist_matrix[current_node][next_node] + self.memo[key]
                    if cost < min_cost:
                        min_cost = cost
                        best_next = next_node

            if best_next is None:
                break

            path.append(best_next)
            visited_set_mask |= (1 << best_next)
            current_node = best_next

        path.append(0)  # return to start
        return path
