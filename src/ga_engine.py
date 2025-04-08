import random
from individual import TriangleIndividual
from fitness import compute_triangle_fitness
from selection import get_selection_method
from termination import check_termination, compute_diversity


class GAEngine:
    def __init__(self, target_image, canvas_size, num_triangles, population_size, num_generations,
                 mutation_rate, crossover_rate, num_mutated_genes,
                 selection_method="tournament", selection_params=None, delta=10,
                 mutation_strategy='single',
                 window_size=10, stagnation_threshold=0.0001, diversity_threshold=0.001):

        self.target_image = target_image
        self.canvas_size = canvas_size
        self.num_triangles = num_triangles
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.delta = delta
        self.mutation_strategy = mutation_strategy
        self.num_mutated_genes = num_mutated_genes
        self.window_size = window_size
        self.stagnation_threshold = stagnation_threshold
        self.diversity_threshold = diversity_threshold

        if selection_params is None:
            selection_params = {}
        self.selection_func = get_selection_method(selection_method, **selection_params)

        self.population = []
        self.best_individual = None
        self.fitness_history = []

    def initialize_population(self):
        """Creates the initial population of triangle individuals."""
        self.population = [
            TriangleIndividual.random_initialize(self.num_triangles, self.canvas_size)
            for _ in range(self.population_size)
        ]

    def evaluate_fitness(self):
        """
        Computes and assigns the fitness for each individual in the population.
        Uses the triangle fitness function (e.g., based on MSE).
        """
        for individual in self.population:
            individual.fitness = compute_triangle_fitness(individual, self.target_image)

    def select_parents(self):
        return self.selection_func(self.population, self.population_size)

    def crossover(self, parent1: TriangleIndividual, parent2: TriangleIndividual):
        """
        Performs a one-point crossover between two parents.
        Returns two new offspring individuals.
        """
        if random.random() > self.crossover_rate:
            # No crossover; return clones of parents
            return parent1.clone(), parent2.clone()

        # Choose a crossover point (ensuring it is within the valid range)
        point = random.randint(1, self.num_triangles - 1)
        child1_triangles = parent1.triangles[:point] + parent2.triangles[point:]
        child2_triangles = parent2.triangles[:point] + parent1.triangles[point:]

        child1 = TriangleIndividual(child1_triangles, self.canvas_size)
        child2 = TriangleIndividual(child2_triangles, self.canvas_size)
        return child1, child2

    def mutate(self, individual: TriangleIndividual):
        """
        Applies mutation to an individual.
        Calls the individual's own mutation method.
        """

        individual.mutate(mutation_rate=self.mutation_rate, delta=self.delta,
                          mutation_strategy=self.mutation_strategy, num_mutated_genes=self.num_mutated_genes)

    def evolve(self):
        """
        Runs the full GA loop:
        - Initializes population
        - Evolves through generations
        - Returns the best individual found
        """
        self.initialize_population()
        self.evaluate_fitness()

        # Use fallback to ensure that individuals with None fitness are treated as very low.
        self.best_individual = max(
            self.population, key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf')).clone()
        self.fitness_history.append(self.best_individual.fitness)

        for gen in range(self.num_generations):
            # Selection: choose parents for the next generation using the configured method
            parents = self.select_parents()
            next_generation = []

            # Crossover and mutation to produce offspring
            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if i + 1 < self.population_size else parents[0]
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                next_generation.extend([child1, child2])

            # Ensure the population size stays consistent
            self.population = next_generation[:self.population_size]
            self.evaluate_fitness()

            current_best = max(
                self.population, key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf'))
            # print("CurrentBest Fitness = ", current_best.fitness, '\n')
            # print('Old best fitness', self.best_individual.fitness, '\n')

            if (current_best.fitness if current_best.fitness is not None else -float('inf')) > (self.best_individual.fitness if self.best_individual.fitness is not None else -float('inf')):
                self.best_individual = current_best.clone()

            self.fitness_history.append(self.best_individual.fitness)
            print(f"Generation {gen+1}: Best Fitness = {self.best_individual.fitness}")

            # if gen > 1:
            #     diversity = compute_diversity(self.population)
            #     print(f"Generation {gen+1}: Diversity = {diversity}")

            #     if check_termination(current_generation=gen+1,
            #                          max_generations=self.num_generations,
            #                          best_fitness_history=self.fitness_history,
            #                          window_size=self.window_size,
            #                          stagnation_threshold=self.stagnation_threshold,
            #                          population=self.population,
            #                          diversity_threshold=self.diversity_threshold):
            #         print("Termination condition met. Ending evolution early.")
            #         break

        return self.best_individual, self.fitness_history
