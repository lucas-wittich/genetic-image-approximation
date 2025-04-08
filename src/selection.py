# Parent selection methods (roulette, tournament, etc.)
import random
import math


def roulette_selection(population, k):
    """
    Fitness proportionate (roulette wheel) selection.
    """

    total_fitness = sum(ind.fitness for ind in population)
    selected = []
    for _ in range(k):
        r = random.uniform(0, total_fitness)
        cumulative = 0
        for ind in population:
            cumulative += ind.fitness
            if cumulative >= r:
                selected.append(ind.clone())
                break
    return selected


def tournament_selection(population, k, tournament_size=5, deterministic=True):
    """
    Tournament selection.

    If deterministic is True, the best individual in the tournament is chosen.
    Otherwise, selection is probabilistic based on fitness.
    """

    selected = []
    for _ in range(k):
        tournament = random.sample(population, tournament_size)
        if deterministic:
            winner = max(tournament, key=lambda ind: ind.fitness)
        else:
            total_fitness = sum(ind.fitness for ind in tournament)
            r = random.uniform(0, total_fitness)
            cumulative = 0
            for ind in tournament:
                cumulative += ind.fitness
                if cumulative >= r:
                    winner = ind
                    break
        selected.append(winner.clone())
    return selected


def ranking_selection(population, k):
    """
    Ranking selection.

    The population is sorted by fitness and individuals are assigned probabilities proportional
    to their rank.
    """

    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    n = len(sorted_pop)
    total_rank = sum(range(1, n+1))
    selected = []
    for _ in range(k):
        r = random.uniform(0, total_rank)
        cumulative = 0
        for rank, ind in enumerate(sorted_pop, start=1):
            cumulative += rank
            if cumulative >= r:
                selected.append(ind.clone())
                break
    return selected


def boltzmann_selection(population, k, temperature=1.0):
    """
    Boltzmann (entropy-based) selection.

    Weights are computed using an exponential function based on fitness and the current temperature.
    """

    weights = [math.exp(ind.fitness / temperature) for ind in population]
    total_weight = sum(weights)
    selected = []
    for _ in range(k):
        r = random.uniform(0, total_weight)
        cumulative = 0
        for ind, w in zip(population, weights):
            cumulative += w
            if cumulative >= r:
                selected.append(ind.clone())
                break
    return selected


def universal_selection(population, k):
    """
    Universal (stochastic remainder) selection.

    Uses equally spaced pointers on the cumulative fitness wheel to select individuals.
    """
    total_fitness = sum(ind.fitness for ind in population)
    start_point = random.uniform(0, total_fitness / k)
    pointers = [start_point + i * total_fitness / k for i in range(k)]
    selected = []
    for pointer in pointers:
        cumulative = 0
        for ind in population:
            cumulative += ind.fitness
            if cumulative >= pointer:
                selected.append(ind.clone())
                break
    return selected


def get_selection_method(method_name, **kwargs):
    """
    Returns a selection function based on the provided method name and parameters.

    Supported method names (case-insensitive):
      - "roulette"
      - "tournament"   (kwargs: tournament_size, deterministic)
      - "ranking"
      - "boltzmann"    (kwargs: temperature)
      - "universal"
    """

    method_name = method_name.lower()
    if method_name == "roulette":
        return lambda population, k: roulette_selection(population, k)
    elif method_name == "tournament":
        return lambda population, k: tournament_selection(population, k,
                                                          tournament_size=kwargs.get("tournament_size", 5),
                                                          deterministic=kwargs.get("deterministic", True))
    elif method_name == "ranking":
        return lambda population, k: ranking_selection(population, k)
    elif method_name == "boltzmann":
        return lambda population, k: boltzmann_selection(population, k, temperature=kwargs.get("temperature", 1.0))
    elif method_name == "universal":
        return lambda population, k: universal_selection(population, k)
    else:
        raise ValueError(f"Unknown selection method: {method_name}")
