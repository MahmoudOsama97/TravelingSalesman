import numpy as np
import random
from utils import total_tour_distance

class TSPSolverGA:
    def __init__(self, cities, population_size=100, generations=200, mutation_rate=0.01):
        """
        cities: np.ndarray of shape (n, 2) with x, y coordinates.
        """
        self.cities = cities
        self.n = len(cities)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def solve(self):
        # Step 1: Initialize population (each tour is a permutation of cities)
        population = [np.random.permutation(self.n) for _ in range(self.population_size)]

        for generation in range(self.generations):
            # Step 2: Evaluate fitness
            fitness_scores = []
            for tour in population:
                tour_list = tour.tolist() if isinstance(tour, np.ndarray) else list(tour)
                distance = total_tour_distance(self.cities, tour_list)
                fitness = 1.0 / (distance + 1e-6)
                fitness_scores.append(fitness)

            # Step 3: Selection (roulette wheel)
            total_fitness = sum(fitness_scores)
            probabilities = [f / total_fitness for f in fitness_scores]
            selected = random.choices(population, weights=probabilities, k=self.population_size)

            # Step 4: Crossover
            next_gen = []
            for _ in range(0, self.population_size, 2):
                p1 = selected[random.randint(0, self.population_size - 1)]
                p2 = selected[random.randint(0, self.population_size - 1)]
                c1, c2 = self._crossover(p1, p2)
                next_gen.extend([c1, c2])

            # Step 5: Mutation
            for i in range(len(next_gen)):
                if random.random() < self.mutation_rate:
                    next_gen[i] = self._mutate(next_gen[i])

            population = next_gen

        # Return the best tour
        best = min(population, key=lambda t: total_tour_distance(self.cities, t))
        return best

    def _crossover(self, p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child1 = [None] * size
        child2 = [None] * size

        child1[a:b] = p1[a:b]
        child2[a:b] = p2[a:b]

        def fill(child, parent):
            pos = b
            for city in parent:
                if city not in child:
                    if pos >= size:
                        pos = 0
                    child[pos] = city
                    pos += 1
            return child

        return fill(child1, p2), fill(child2, p1)

    def _mutate(self, tour):
        a, b = random.sample(range(len(tour)), 2)
        tour[a], tour[b] = tour[b], tour[a]
        return tour
