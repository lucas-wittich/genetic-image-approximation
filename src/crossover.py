# Crossover methods (1-point, 2-point, uniform, etc.)
import copy
import random
from individual import TriangleIndividual


def one_point_crossover(parent1, parent2):
    """
    Perform one-point crossover on two TriangleIndividuals.
    """
    point = random.randint(1, len(parent1.triangles) - 1)

    child1_triangles = parent1.triangles[:point] + parent2.triangles[point:]
    child2_triangles = parent2.triangles[:point] + parent1.triangles[point:]

    child1 = TriangleIndividual(child1_triangles, parent1.canvas_size)
    child2 = TriangleIndividual(child2_triangles, parent1.canvas_size)

    return child1, child2


def two_point_crossover(parent1, parent2):
    """
    Perform two-point crossover on two TriangleIndividuals.
    """
    length = len(parent1.triangles)
    p1, p2 = sorted(random.sample(range(1, length), 2))

    child1_triangles = (
        parent1.triangles[:p1]
        + parent2.triangles[p1:p2]
        + parent1.triangles[p2:]
    )
    child2_triangles = (
        parent2.triangles[:p1]
        + parent1.triangles[p1:p2]
        + parent2.triangles[p2:]
    )

    child1 = TriangleIndividual(child1_triangles, parent1.canvas_size)
    child2 = TriangleIndividual(child2_triangles, parent1.canvas_size)

    return child1, child2


def uniform_crossover(parent1, parent2, swap_prob=0.5):
    """
    Perform uniform crossover on two TriangleIndividuals.
    """
    child1_triangles = []
    child2_triangles = []

    for t1, t2 in zip(parent1.triangles, parent2.triangles):
        if random.random() < swap_prob:
            child1_triangles.append(copy.deepcopy(t2))
            child2_triangles.append(copy.deepcopy(t1))
        else:
            child1_triangles.append(copy.deepcopy(t1))
            child2_triangles.append(copy.deepcopy(t2))

    child1 = TriangleIndividual(child1_triangles, parent1.canvas_size)
    child2 = TriangleIndividual(child2_triangles, parent1.canvas_size)

    return child1, child2
