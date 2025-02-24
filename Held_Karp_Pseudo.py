def held_karp(cities, cost_matrix):
    n = len(cities)
    # Initialize the DP table
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Base case: starting at city 0

    # Iterate over all subsets
    for subset_size in range(2, n + 1):
        for subset in combinations(range(1, n), subset_size - 1):
            subset_mask = 1 << 0  # Always include the starting city
            for city in subset:
                subset_mask |= 1 << city
            # Iterate over each city in the subset
            for next_city in subset:
                prev_mask = subset_mask & ~(1 << next_city)
                # Iterate over all possible previous cities
                for prev_city in subset:
                    if prev_city == next_city:
                        continue
                    dp[subset_mask][next_city] = min(
                        dp[subset_mask][next_city],
                        dp[prev_mask][prev_city] + cost_matrix[prev_city][next_city]
                    )

    # Compute the final result
    final_mask = (1 << n) - 1
    min_cost = float('inf')
    for last_city in range(1, n):
        min_cost = min(min_cost, dp[final_mask][last_city] + cost_matrix[last_city][0])

    return min_cost