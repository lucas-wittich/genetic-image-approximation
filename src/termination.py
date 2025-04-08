import math


def stop_after_max_generations(current_generation, max_generations):
    """
    Terminate if the current generation count is greater than or equal to max_generations.
    """
    return current_generation >= max_generations


def stop_if_stagnant(best_fitness_history, window_size, stagnation_threshold):
    """
    Terminate if the improvement in best fitness over the last 'window_size' generations
    is less than 'stagnation_threshold'.
    """
    if len(best_fitness_history) < window_size:
        return False  # Not enough data to decide
    recent_fitness = best_fitness_history[-window_size:]
    improvement = max(recent_fitness) - min(recent_fitness)
    return improvement < stagnation_threshold


def compute_diversity(population):
    """
    Compute a simple diversity metric for the population based on the standard deviation
    of the fitness values. A low standard deviation suggests that the population may have converged.
    """

    fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
    if not fitnesses:
        return 0.0
    mean = sum(fitnesses) / len(fitnesses)
    variance = sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)
    diversity = math.sqrt(variance)
    return diversity


def structure_convergence(population, diversity_threshold):
    """
    Terminate if the computed diversity of the population falls below the specified threshold.
    """
    diversity = compute_diversity(population)
    return diversity < diversity_threshold


def check_termination(current_generation, max_generations, best_fitness_history,
                      window_size, stagnation_threshold, population, diversity_threshold):
    """
    Returns True if any termination condition is met:
      1. Maximum generations reached.
      2. Improvement over the last 'window_size' generations is below threshold.
      3. Population diversity is below threshold.
    """
    if stop_after_max_generations(current_generation, max_generations):
        return True
    if stop_if_stagnant(best_fitness_history, window_size, stagnation_threshold):
        return True
    if structure_convergence(population, diversity_threshold):
        return True
    return False
