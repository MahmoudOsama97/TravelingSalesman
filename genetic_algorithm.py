import random
from utils import total_tour_distance


class TSPSolverGA:
    def __init__(self, dist_matrix, population_size=50, generations=200, mutation_rate=0.02):
        self.dist_matrix = dist_matrix
        self.n = len(dist_matrix)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def solve(self):
        """
        Run the genetic algorithm and return the best found tour and its cost.
        """
        # Generate initial population
        population = [self._random_tour() for _ in range(self.population_size)]
        best_tour = None
        best_cost = float("inf")

        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = [1.0 / (total_tour_distance(tour, self.dist_matrix) + 1e-6) for tour in population]

            # Track the best in this generation
            for i, tour in enumerate(population):
                cost = total_tour_distance(tour, self.dist_matrix)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour[:]

            # Selection & Reproduction
            new_population = []
            for _ in range(self.population_size // 2):
                # Parent selection (tournament or roulette)
                parent1 = self._select_tournament(population, fitness_scores)
                parent2 = self._select_tournament(population, fitness_scores)

                # Crossover
                child1, child2 = self._order_crossover(parent1, parent2)

                # Mutation
                if random.random() < self.mutation_rate:
                    self._swap_mutation(child1)
                if random.random() < self.mutation_rate:
                    self._swap_mutation(child2)

                new_population.append(child1)
                new_population.append(child2)

            population = new_population

        return best_tour, best_cost

    def _random_tour(self):
        tour = list(range(self.n))
        random.shuffle(tour)
        return tour

    def _select_tournament(self, population, fitness_scores, k=3):
        """
        Tournament selection: pick k random individuals, return the best.
        """
        selected = random.sample(list(zip(population, fitness_scores)), k)
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[0][0]

    def _order_crossover(self, p1, p2):
        """
        Order crossover (OX): preserves the relative ordering of cities.
        """
        size = len(p1)
        c1, c2 = [-1] * size, [-1] * size

        start, end = sorted([random.randrange(size) for _ in range(2)])

        # Copy the segment from parent1
        for i in range(start, end + 1):
            c1[i] = p1[i]
        # Fill in the rest from parent2 in order
        pos = (end + 1) % size
        for city in p2:
            if city not in c1:
                c1[pos] = city
                pos = (pos + 1) % size

        # Repeat for child2
        start, end = sorted([random.randrange(size) for _ in range(2)])
        for i in range(start, end + 1):
            c2[i] = p2[i]
        pos = (end + 1) % size
        for city in p1:
            if city not in c2:
                c2[pos] = city
                pos = (pos + 1) % size

        return c1, c2

    def _swap_mutation(self, tour):
        """
        Swap two cities in the tour.
        """
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
