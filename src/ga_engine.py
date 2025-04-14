import random
from individual import TriangleIndividual
from fitness import compute_triangle_fitness
from selection import get_selection_method
from termination import check_termination, compute_diversity

import random
import copy
import numpy as np
from individual import TriangleIndividual
from selection import get_selection_method


class GAEngine:

    def __init__(self, target_image, canvas_size, num_triangles, population_size,
                 num_generations, mutation_rate, crossover_rate, num_mutated_genes,
                 selection_method, selection_params, mutation_strategy, termination_params,
                 elitism_rate=0.1):

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
        self.termination_params = termination_params  # set this from config if needed
        self.elitism_rate = elitism_rate
        self.population = []
        self.best_individual = None

    def initialize_population(self):
        self.population = [
            TriangleIndividual.random_initialize(self.num_triangles, self.canvas_size)
            for _ in range(self.population_size)
        ]

    def evaluate_fitness(self):
        for individual in self.population:
            individual.fitness = compute_triangle_fitness(individual, self.target_image)

    def select_parents(self):
        selection_fn = get_selection_method(self.selection_method, **self.selection_params)
        return selection_fn(self.population, self.population_size)

    def one_point_crossover(self, parent1, parent2):
        point = random.randint(1, self.num_triangles - 1)
        child1_triangles = parent1.triangles[:point] + parent2.triangles[point:]
        child2_triangles = parent2.triangles[:point] + parent1.triangles[point:]
        child1 = TriangleIndividual(child1_triangles, self.canvas_size)
        child2 = TriangleIndividual(child2_triangles, self.canvas_size)
        return child1, child2

    def evolve(self):
        self.initialize_population()
        self.evaluate_fitness()

        best_fitness_history = []
        avg_fitness_history = []
        diversity_history = []
        max_generations = self.num_generations
        window_size = self.termination_params.get('window_size', 10)
        stagnation_threshold = self.termination_params.get('stagnation_threshold', 0.001)
        diversity_threshold = self.termination_params.get('diversity_threshold', 0.001)

        for gen in range(self.num_generations):

            snapshots = []
            snapshot_interval = self.termination_params.get("snapshot_interval", 100)

            self.population.sort(key=lambda ind: ind.fitness)
            elite_count = max(1, int(self.elitism_rate * self.population_size))
            elites = [ind.clone() for ind in self.population[:elite_count]]

            parents = self.select_parents()
            next_generation = []

            for i in range(0, self.population_size - elite_count, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

                if random.random() < self.crossover_rate:
                    child1, child2 = self.one_point_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.clone(), parent2.clone()

                child1.mutate(
                    mutation_rate=self.mutation_rate,
                    mutation_strategy=self.mutation_strategy,
                    num_mutated_genes=self.num_mutated_genes
                )
                child2.mutate(
                    mutation_rate=self.mutation_rate,
                    mutation_strategy=self.mutation_strategy,
                    num_mutated_genes=self.num_mutated_genes
                )
                next_generation.extend([child1, child2])

            self.population = elites + next_generation[:self.population_size - elite_count]
            self.evaluate_fitness()

            self.best_individual = min(self.population, key=lambda ind: ind.fitness)
            best_fitness = self.best_individual.fitness
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            diversity = compute_diversity(self.population)

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            diversity_history.append(diversity)

            print(f"Gen {gen+1} | Best: {best_fitness:.6f} | Avg: {avg_fitness:.6f} | Diversity: {diversity:.4f}")
            if (gen + 1) % snapshot_interval == 0:
                snapshots.append(self.best_individual.render())
            if check_termination(gen, max_generations, best_fitness_history,
                                 window_size, stagnation_threshold,
                                 self.population, diversity_threshold):
                break

        return self.best_individual, {
            "snapshots": snapshots,
            "best_fitness": best_fitness_history,
            "avg_fitness": avg_fitness_history,
            "diversity": diversity_history
        }
