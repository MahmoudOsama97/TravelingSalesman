import math


class TSPSolverHeldKarp:
    def __init__(self, dist_matrix):
        self.dist_matrix = dist_matrix
        self.n = len(dist_matrix)
        self.memo = {}  # Memoization table to store results of subproblems
        self.best_tour = None
        self.best_cost = math.inf

    def solve(self):
        """
        Public method to solve TSP using Held-Karp algorithm and return the best tour and cost.
        """
        start_node = 0
        # Initialize the set of visited cities (excluding start node) as a bitmask
        # All cities except the starting city (city 0) are in the initial set
        initial_set_mask = (1 << self.n) - 1  # e.g., for n=4, mask = 1111 (binary) = 15 (decimal)
        initial_set_mask &= ~(1 << start_node)  # Remove start node from the set

        self.best_cost = self._held_karp_recursive(start_node, initial_set_mask)

        # Reconstruct the tour (optional, if you need the path)
        if self.best_cost != math.inf:
            self.best_tour = self._reconstruct_tour(start_node, initial_set_mask)
        else:
            self.best_tour = None

        return self.best_tour, self.best_cost

    def _held_karp_recursive(self, last_node, visited_set_mask):
        # print(f"HK_recursive: last_node={last_node}, visited_set_mask={visited_set_mask}")
        if visited_set_mask == 0:
            base_cost = self.dist_matrix[last_node][0]
            # print(f"HK_recursive BASE CASE: key={(last_node, visited_set_mask)}, cost={base_cost}") # Base case print
            self.memo[(last_node, visited_set_mask)] = base_cost
            return base_cost

        if (last_node, visited_set_mask) in self.memo:
            memo_val = self.memo[(last_node, visited_set_mask)]
            #   print(f"HK_recursive MEMO HIT: key={(last_node, visited_set_mask)}, value={memo_val}") # Memo hit print
            return memo_val

        min_cost = math.inf
        for next_node in range(self.n):
            if (visited_set_mask >> next_node) & 1:
                next_visited_set_mask = visited_set_mask & ~(1 << next_node)
                cost = self.dist_matrix[last_node][next_node] + self._held_karp_recursive(
                    next_node, next_visited_set_mask
                )
                if cost < min_cost:
                    min_cost = cost

        # print(f"HK_recursive MEMO STORE: key={(last_node, visited_set_mask)}, value={min_cost}") # Memo store print
        self.memo[(last_node, visited_set_mask)] = min_cost
        return min_cost

    def _reconstruct_tour(self, start_node, initial_set_mask):
        tour = [start_node]
        current_node = start_node
        current_set_mask = initial_set_mask

        while current_set_mask > 0:
            best_next_node = -1
            min_cost_to_next_node = math.inf

            for next_node in range(self.n):
                if (current_set_mask >> next_node) & 1:
                    next_set_mask = current_set_mask & ~(1 << next_node)
                    # print(f"Reconstruct: current_node={current_node}, next_node={next_node}, next_set_mask={next_set_mask}") # Reconstruction print
                    cost = (
                        self.dist_matrix[current_node][next_node] + self.memo[(next_node, next_set_mask)]
                    )  # KeyError here
                    if cost < min_cost_to_next_node:
                        min_cost_to_next_node = cost
                        best_next_node = next_node

            tour.append(best_next_node)
            current_node = best_next_node
            current_set_mask &= ~(1 << best_next_node)

        return tour
