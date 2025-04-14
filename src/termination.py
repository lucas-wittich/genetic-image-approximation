import math


def stop_after_max_generations(current_generation, max_generations):
    """Terminate if the max generation count has been reached."""
    return current_generation >= max_generations


def stop_if_stagnant(best_fitness_history, window_size, stagnation_threshold):
    """Terminate if improvement over the last window_size generations is too small."""
    if len(best_fitness_history) < window_size:
        return False
    recent_fitness = best_fitness_history[-window_size:]
    improvement = max(recent_fitness) - min(recent_fitness)
    return improvement < stagnation_threshold


def compute_diversity(population):
    """Compute population diversity as std. deviation of fitness values."""
    fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
    if not fitnesses:
        return 0.0
    mean = sum(fitnesses) / len(fitnesses)
    variance = sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)
    return math.sqrt(variance)


def structure_convergence(population, diversity_threshold):
    """Terminate if population diversity falls below the threshold."""
    diversity = compute_diversity(population)
    return diversity < diversity_threshold


def check_termination(
    current_generation,
    max_generations,
    best_fitness_history,
    window_size,
    stagnation_threshold,
    population,
    diversity_threshold
):
    """
    Aggregate termination condition:
    - max_generations: stop if reached
    - stagnation: stop if best fitness hasn't improved over the last window_size
    - structural convergence: stop if population diversity is too low
    """
    if stop_after_max_generations(current_generation, max_generations):
        print(f"Terminating: generation limit ({current_generation} >= {max_generations})")
        return True
    if stop_if_stagnant(best_fitness_history, window_size, stagnation_threshold):
        print(f"Terminating: stagnation over last {window_size} generations (< {stagnation_threshold})")
        return True
    if structure_convergence(population, diversity_threshold):
        print(f"Terminating: population diversity dropped below {diversity_threshold}")
        return True
    return False
