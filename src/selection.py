# Parent selection methods (roulette, tournament, etc.)
import random
import math


def invert_fitnesses(population):
    fitnesses = [ind.fitness for ind in population]

    min_f = min(fitnesses)
    if min_f <= 0:
        fitnesses = [f - min_f + 1e-8 for f in fitnesses]

    return fitnesses


def roulette_selection(population, k):
    """Fitness-proportionate (roulette wheel) selection for minimization problems."""
    weights = invert_fitnesses(population)
    total = sum(weights)
    selected = []
    for _ in range(k):
        r = random.uniform(0, total)
        cumulative = 0
        for ind, w in zip(population, weights):
            cumulative += w
            if cumulative >= r:
                selected.append(ind.clone())
                break
    return selected


def tournament_selection(population, k, tournament_size=5, deterministic=True):
    """Tournament selection (best of N or probabilistic)."""
    selected = []
    for _ in range(k):
        tournament = random.sample(population, tournament_size)
        if deterministic:
            winner = max(tournament, key=lambda ind: ind.fitness)
        else:
            scores = invert_fitnesses(tournament)
            total = sum(scores)
            r = random.uniform(0, total)
            cumulative = 0
            for ind, s in zip(tournament, scores):
                cumulative += s
                if cumulative >= r:
                    winner = ind
                    break
        selected.append(winner.clone())
    return selected


def ranking_selection(population, k):
    """Ranking selection assigns probability based on sorted rank."""
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    n = len(sorted_pop)
    total_rank = sum(range(1, n + 1))
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
    """Boltzmann selection (temperature-based softmax over inverted fitness)."""
    scores = invert_fitnesses(population)
    weights = [math.exp(score / temperature) for score in scores]
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
    """Universal selection: evenly spaced roulette pointers."""
    scores = invert_fitnesses(population)
    total_score = sum(scores)
    step = total_score / k
    start = random.uniform(0, step)
    points = [start + i * step for i in range(k)]
    selected = []
    for point in points:
        cumulative = 0
        for ind, s in zip(population, scores):
            cumulative += s
            if cumulative >= point:
                selected.append(ind.clone())
                break
    return selected


def get_selection_method(method_name, **kwargs):
    """Return configured selection method with kwargs injected."""
    method_name = method_name.lower()
    if method_name == "roulette":
        return lambda pop, k: roulette_selection(pop, k)
    elif method_name == "tournament":
        return lambda pop, k: tournament_selection(pop, k,
                                                   tournament_size=kwargs.get("tournament_size", 5),
                                                   deterministic=kwargs.get("deterministic", True))
    elif method_name == "ranking":
        return lambda pop, k: ranking_selection(pop, k)
    elif method_name == "boltzmann":
        return lambda pop, k: boltzmann_selection(pop, k,
                                                  temperature=kwargs.get("temperature", 1.0))
    elif method_name == "universal":
        return lambda pop, k: universal_selection(pop, k)
    else:
        raise ValueError(f"Unknown selection method: {method_name}")
