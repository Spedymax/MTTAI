import numpy as np
from typing import Tuple, List


class GeneticOptimizer:
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 elite_size: int = 2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def initialize_population(self, bounds: List[Tuple[float, float]], size: int) -> np.ndarray:
        """Initialize random population within given bounds"""
        dim = len(bounds)
        population = np.zeros((size, dim))
        for i in range(dim):
            population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], size)
        return population

    def fitness_single_var(self, x: float) -> float:
        """Fitness function for f(x) = cos(x)/x - sin(x)/x^2"""
        if abs(x) < 1e-10:  # Avoid division by zero
            return float('inf')
        return np.cos(x) / x - np.sin(x) / (x ** 2)

    def fitness_two_var(self, x: float, y: float) -> float:
        """Fitness function for z = sin(x/2) + y*sin(x)"""
        return np.sin(x / 2) + y * np.sin(x)

    def selection(self, population: np.ndarray, fitness: np.ndarray, minimize: bool = True) -> np.ndarray:
        """Tournament selection"""
        new_population = np.zeros_like(population)

        # Elitism
        if minimize:
            elite_idx = np.argsort(fitness)[:self.elite_size]
        else:
            elite_idx = np.argsort(fitness)[-self.elite_size:]

        new_population[:self.elite_size] = population[elite_idx]

        # Tournament selection for the rest
        for i in range(self.elite_size, len(population)):
            tournament_idx = np.random.choice(len(population), 3)
            if minimize:
                winner_idx = tournament_idx[np.argmin(fitness[tournament_idx])]
            else:
                winner_idx = tournament_idx[np.argmax(fitness[tournament_idx])]
            new_population[i] = population[winner_idx]

        return new_population

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform arithmetic crossover"""
        alpha = np.random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def mutation(self, individual: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Perform gaussian mutation"""
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                sigma = (bounds[i][1] - bounds[i][0]) * 0.1
                individual[i] += np.random.normal(0, sigma)
                individual[i] = np.clip(individual[i], bounds[i][0], bounds[i][1])
        return individual

    def optimize_single_var(self, bounds: Tuple[float, float]) -> Tuple[float, float]:
        """Optimize single variable function"""
        population = self.initialize_population([bounds], self.population_size)
        best_fitness = float('inf')
        best_solution = None

        for generation in range(self.generations):
            fitness = np.array([self.fitness_single_var(x[0]) for x in population])

            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_solution = population[np.argmin(fitness)]

            population = self.selection(population, fitness)

            # Crossover and mutation
            for i in range(0, self.population_size - 1, 2):
                if np.random.random() < 0.8:  # crossover probability
                    population[i], population[i + 1] = self.crossover(population[i], population[i + 1])

                population[i] = self.mutation(population[i], [bounds])
                population[i + 1] = self.mutation(population[i + 1], [bounds])

        return best_solution[0], best_fitness

    def optimize_two_var(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """Optimize two variable function"""
        population = self.initialize_population(bounds, self.population_size)
        best_fitness = float('-inf')
        best_solution = None

        for generation in range(self.generations):
            fitness = np.array([self.fitness_two_var(x[0], x[1]) for x in population])

            if np.max(fitness) > best_fitness:
                best_fitness = np.max(fitness)
                best_solution = population[np.argmax(fitness)]

            population = self.selection(population, fitness, minimize=False)

            # Crossover and mutation
            for i in range(0, self.population_size - 1, 2):
                if np.random.random() < 0.8:  # crossover probability
                    population[i], population[i + 1] = self.crossover(population[i], population[i + 1])

                population[i] = self.mutation(population[i], bounds)
                population[i + 1] = self.mutation(population[i + 1], bounds)

        return best_solution, best_fitness


# Example usage
if __name__ == "__main__":
    optimizer = GeneticOptimizer(population_size=200, generations=100)

    # Task 1: Find minimum of single variable function
    x_bounds = (-10, 10)
    x_min, min_value = optimizer.optimize_single_var(x_bounds)
    print(f"\nTask 1 - Minimum found:")
    print(f"x = {x_min:.6f}")
    print(f"f(x) = {min_value:.6f}")

    # Task 2: Find maximum of two variable function
    xy_bounds = [(-10, 10), (-10, 10)]  # bounds for x and y
    (x_max, y_max), max_value = optimizer.optimize_two_var(xy_bounds)
    print(f"\nTask 2 - Maximum found:")
    print(f"x = {x_max:.6f}")
    print(f"y = {y_max:.6f}")
    print(f"z = {max_value:.6f}")