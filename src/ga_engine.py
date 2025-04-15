
import random
import numpy as np
from individual import TriangleIndividual
from fitness import compute_triangle_fitness
from selection import get_selection_method
from termination import check_termination, compute_diversity
from crossover import one_point_crossover, two_point_crossover, uniform_crossover


def get_crossover_function(name):
    name = name.lower()
    if name == "one_point":
        return one_point_crossover
    elif name == "two_point":
        return two_point_crossover
    elif name == "uniform":
        return uniform_crossover
    else:
        raise ValueError(f"Unsupported crossover method: {name}")


class GAEngine:

    def __init__(self, target_image, canvas_size, num_triangles, population_size,
                 num_generations, mutation_rate, crossover_rate, num_mutated_genes,
                 selection_method, selection_params, mutation_strategy, termination_params,
                 delta=10, young_bias_ratio=0.8,
                 crossover_method="one_point", elitism_rate=0.1, generation_approach="traditional"):

        self.target_image = target_image
        self.canvas_size = canvas_size
        self.num_triangles = num_triangles
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_mutated_genes = num_mutated_genes
        self.selection_method = selection_method
        self.selection_params = selection_params
        self.mutation_strategy = mutation_strategy
        self.termination_params = termination_params
        self.elitism_rate = elitism_rate
        self.crossover_func = get_crossover_function(crossover_method)
        self.generation_approach = generation_approach
        self.delta = delta
        self.young_bias_ratio = young_bias_ratio

        self.population = []
        self.best_individual = None

    def initialize_population(self):
        self.population = [
            TriangleIndividual.random_initialize(self.num_triangles, self.canvas_size)
            for _ in range(self.population_size)
        ]

    def evaluate_fitness(self, population=None):
        if population == None:
            population = self.population
        for individual in population:
            individual.fitness = compute_triangle_fitness(individual, self.target_image)

    def select_parents(self):
        selection_fn = get_selection_method(self.selection_method, **self.selection_params)
        return selection_fn(self.population, self.population_size)

    def evolve(self):
        self.initialize_population()
        self.evaluate_fitness()

        best_fitness_history = []
        avg_fitness_history = []
        diversity_history = []
        normalized_fitness_history = []
        snapshots = []
        snapshot_interval = self.termination_params.get("snapshot_interval", 100)

        for gen in range(self.num_generations):
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            elite_count = max(1, int(self.elitism_rate * self.population_size))
            elites = [ind.clone() for ind in self.population[:elite_count]]

            parents = self.select_parents()
            next_generation = []

            for i in range(0, self.population_size - elite_count, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover_func(parent1, parent2)
                else:
                    child1, child2 = parent1.clone(), parent2.clone()

                child1.mutate(mutation_rate=self.mutation_rate, delta=self.delta,
                              mutation_strategy=self.mutation_strategy, num_mutated_genes=self.num_mutated_genes)
                child2.mutate(mutation_rate=self.mutation_rate, delta=self.delta, mutation_strategy=self.mutation_strategy,
                              num_mutated_genes=self.num_mutated_genes)
                next_generation.extend([child1, child2])

            if self.generation_approach == "young_bias":
                # Proportional bias: keep more offspring, fewer elites
                offspring_count = int(self.population_size * self.young_bias_ratio)
                elite_count = self.population_size - offspring_count
                self.evaluate_fitness(population=next_generation)
                combined_pool = next_generation + elites
                combined_pool.sort(key=lambda ind: ind.fitness, reverse=True)
                self.population = combined_pool[:self.population_size]
            else:
                self.population = elites + next_generation[:self.population_size - elite_count]

            self.evaluate_fitness()

            self.best_individual = max(self.population, key=lambda ind: ind.fitness)
            best_fitness = self.best_individual.fitness
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            diversity = compute_diversity(self.population)
            if gen == 0:
                initial_fitness = best_fitness

            normalized_fitness = (best_fitness / initial_fitness)

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            diversity_history.append(diversity)
            normalized_fitness_history.append(normalized_fitness)

            print(f"Gen {gen+1} | Best: {best_fitness:.6f} | Avg: {avg_fitness:.6f} | Diversity: {diversity:.4f}")
            if (gen + 1) % snapshot_interval == 0:
                snapshots.append(self.best_individual.render())

            if check_termination(gen, self.num_generations, best_fitness_history,
                                 self.termination_params.get("window_size", 10),
                                 self.termination_params.get("stagnation_threshold", 0.001),
                                 self.population,
                                 self.termination_params.get("diversity_threshold", 0.001)):
                break

        return self.best_individual, {
            "best_fitness": best_fitness_history,
            "avg_fitness": avg_fitness_history,
            "diversity": diversity_history,
            "normalized_fitness": normalized_fitness_history,
            "snapshots": snapshots
        }
