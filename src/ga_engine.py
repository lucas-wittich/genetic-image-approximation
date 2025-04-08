import random
from individual import TriangleIndividual
from fitness import compute_triangle_fitness
from selection import get_selection_method


class GAEngine:
    def __init__(self, target_image, canvas_size, num_triangles, population_size, num_generations,
                 mutation_rate, crossover_rate, selection_method="tournament", selection_params=None):
        """
        Initializes the GA engine with configuration parameters.

        Parameters:
        - target_image: PIL.Image object of the target image.
        - canvas_size: Tuple (width, height) for the drawing canvas.
        - num_triangles: Number of triangles per individual.
        - population_size: Number of individuals in the population.
        - num_generations: Number of generations to run.
        - mutation_rate: Probability of mutation for each attribute.
        - crossover_rate: Probability of applying crossover.
        - selection_method: String identifier for the selection method (e.g., "tournament", "roulette").
        - selection_params: Dictionary with parameters specific to the selection method.
        """
        self.target_image = target_image
        self.canvas_size = canvas_size
        self.num_triangles = num_triangles
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

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
        """
        Uses the configured selection function to choose parents.
        Returns a new list of individuals (clones) selected from the population.
        """
        return self.selection_func(self.population, self.population_size)

    def crossover(self, parent1, parent2):
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

    def mutate(self, individual):
        """
        Applies mutation to an individual.
        Calls the individual's own mutation method.
        """

        individual.mutate(mutation_rate=self.mutation_rate)

    def evolve(self):
        """
        Runs the full GA loop:
          - Initializes population
          - Evolves through generations
          - Returns the best individual found
        """

        self.initialize_population()
        self.evaluate_fitness()
        self.best_individual = max(self.population, key=lambda ind: ind.fitness).clone()
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

            # Update the best individual if a better one is found
            current_best = max(self.population, key=lambda ind: ind.fitness)
            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.clone()

            self.fitness_history.append(self.best_individual.fitness)
            print(f"Generation {gen+1}: Best Fitness = {self.best_individual.fitness}")

            # Optional: additional termination conditions can be added here

        return self.best_individual
