import os
import random
import numpy as np
from multiprocessing import Pool
from individual import TriangleIndividual
from fitness import compute_triangle_fitness, init_target
from selection import get_selection_method
from termination import check_termination, compute_diversity
from crossover import one_point_crossover, two_point_crossover, uniform_crossover

# Global constant for multiprocessing
# _WORKER_TARGET = None


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


def _init_worker(target_img):
    # global _WORKER_TARGET
    init_target(target_img)
    # _WORKER_TARGET = target_img


def _eval_individual(individual):
    from fitness import compute_triangle_fitness
    return compute_triangle_fitness(individual)


class GAEngine:

    def __init__(self, target_image, canvas_size, num_triangles, population_size,
                 num_generations, mutation_rate, crossover_rate, num_mutated_genes,
                 selection_method, selection_params, mutation_strategy, termination_params,
                 delta=10, young_bias_ratio=0.8,
                 crossover_method="one_point", elitism_rate=0.1, generation_approach="traditional"):

        self.target_image = target_image
        init_target(self.target_image)

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

        self.generation_approach = generation_approach
        self.delta = delta
        self.young_bias_ratio = young_bias_ratio
        if isinstance(crossover_method, list):
            self.crossover_funcs = [get_crossover_function(item) for item in crossover_method]
        else:
            self.crossover_funcs = [get_crossover_function(crossover_method)]
        self._pool = Pool(processes=os.cpu_count(),
                          initializer=_init_worker,
                          initargs=(self.target_image,))

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
            individual.fitness = compute_triangle_fitness(individual)

    def multithread_evaluate_fitness(self, population=None):
        if population == None:
            population = self.population

        fitnesses = self._pool.map(_eval_individual, population)

        for ind, fit in zip(population, fitnesses):
            ind.fitness = fit

    def select_parents(self):
        selection_fn = get_selection_method(self.selection_method, **self.selection_params)
        return selection_fn(self.population, self.population_size)

    def evolve(self):
        self.initialize_population()
        self.multithread_evaluate_fitness()

        best_fitness_history = []
        avg_fitness_history = []
        diversity_history = []
        normalized_fitness_history = []
        snapshots = []
        snapshot_interval = self.termination_params.get("snapshot_interval", 100)

        base_mutation = self.mutation_rate
        min_mutation = 0.05  # floor at 5%

        for gen in range(self.num_generations):
            # frac = gen / max(1, self.num_generations - 1)
            # current_mutation = base_mutation * (1 - frac) + min_mutation * frac
            current_mutation = base_mutation

            if self.generation_approach == "young_bias":
                offspring_count = int(self.population_size * self.young_bias_ratio)
                elite_count = self.population_size - offspring_count
            else:
                elite_count = max(1, int(self.elitism_rate * self.population_size))
                offspring_count = self.population_size - elite_count

            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            elites = [ind.clone() for ind in self.population[:elite_count]]

            parents = self.select_parents()

            children = []
            while len(children) < offspring_count:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                if random.random() < self.crossover_rate:
                    fn = random.choice(self.crossover_funcs)
                    c1, c2 = fn(p1, p2)
                else:
                    c1, c2 = p1.clone(), p2.clone()

                c1.mutate(mutation_rate=current_mutation,
                          delta=self.delta,
                          mutation_strategy=self.mutation_strategy,
                          num_mutated_genes=self.num_mutated_genes)
                c2.mutate(mutation_rate=current_mutation,
                          delta=self.delta,
                          mutation_strategy=self.mutation_strategy,
                          num_mutated_genes=self.num_mutated_genes)

                children.extend([c1, c2])

            children = children[:offspring_count]

            new_population = elites + children

            self.multithread_evaluate_fitness(population=children)
            self.population = new_population

            self.best_individual = max(self.population, key=lambda ind: ind.fitness)
            best_f = self.best_individual.fitness
            fits = np.fromiter((ind.fitness for ind in self.population), float)
            avg_f = fits.mean()
            diversity = fits.std()

            if gen == 0:
                initial_f = best_f

            best_fitness_history.append(best_f)
            avg_fitness_history.append(avg_f)
            diversity_history.append(diversity)
            normalized_fitness_history.append(best_f / initial_f)

            if (gen + 1) % snapshot_interval == 0:
                snapshots.append(self.best_individual.render())
            print(f"Gen {gen+1} | Best: {best_f:.6f} | Avg: {avg_f:.6f} | Div: {diversity:.4f}")

            if check_termination(gen, self.num_generations, best_fitness_history,
                                 self.termination_params.get("window_size", 10),
                                 self.termination_params.get("stagnation_threshold", 0.001),
                                 self.population,
                                 self.termination_params.get("diversity_threshold", 0.001)):
                break

        self._pool.close()
        self._pool.join()

        return self.best_individual, {
            "best_fitness": best_fitness_history,
            "avg_fitness": avg_fitness_history,
            "diversity": diversity_history,
            "normalized_fitness": normalized_fitness_history,
            "snapshots": snapshots
        }
