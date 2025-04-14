
import random
import numpy as np

# Gene groupings
coord_keys = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3']
color_keys = [("color", i) for i in range(4)]
all_genes = color_keys + coord_keys


def delta_mutation(gene_value, delta, max_delta=20):
    """Apply a bounded delta mutation to a single gene value."""
    adjustment = np.clip(random.uniform(-delta, delta), -max_delta, max_delta)
    return gene_value + adjustment


def clamp_color_value(index, value):
    """Clamp RGB values to [0,180], alpha to [30,150]."""
    return int(np.clip(value, 0, 180)) if index < 3 else int(np.clip(value, 30, 150))


def mutate_coords(triangle, key, delta, canvas_size):
    """Mutate and clamp a coordinate gene (x or y)."""
    width, height = canvas_size
    new_value = delta_mutation(triangle[key], delta)
    max_val = width if key.startswith("x") else height
    triangle[key] = int(np.clip(new_value, 0, max_val))


def mutate_color(triangle, index, delta):
    """Mutate and clamp a color channel."""
    color = list(triangle["color"])
    new_value = delta_mutation(color[index], delta)
    color[index] = clamp_color_value(index, new_value)
    triangle["color"] = tuple(color)


def single_gene_mutation(triangle, mutation_rate, delta, canvas_size):
    """Mutate one randomly chosen gene (coordinate or color) if triggered."""
    if random.random() < mutation_rate:
        gene = random.choice(all_genes)
        if isinstance(gene, str):
            mutate_coords(triangle, gene, delta, canvas_size)
        else:
            mutate_color(triangle, gene[1], delta)
    return triangle


def limited_multi_gene_mutation(triangle, delta, canvas_size, mutation_rate, num_mutated_genes=None):
    """Mutate a limited number of randomly chosen genes."""
    if random.random() < mutation_rate:
        M = len(all_genes)
        num_mutated_genes = num_mutated_genes or random.randint(1, M)
        for gene in random.sample(all_genes, num_mutated_genes):
            if isinstance(gene, str):
                mutate_coords(triangle, gene, delta, canvas_size)
            else:
                mutate_color(triangle, gene[1], delta)
    return triangle


def uniform_multi_gene_mutation(triangle, mutation_rate, delta, canvas_size):
    """Mutate each gene independently with mutation_rate probability."""
    for key in coord_keys:
        if random.random() < mutation_rate:
            mutate_coords(triangle, key, delta, canvas_size)
    for key, index in color_keys:
        if random.random() < mutation_rate:
            mutate_color(triangle, index, delta)
    return triangle


def complete_mutation(triangle, mutation_rate, delta, canvas_size):
    """Mutate all genes if mutation triggers."""
    if random.random() < mutation_rate:
        for key in coord_keys:
            mutate_coords(triangle, key, delta, canvas_size)
        for _, index in color_keys:
            mutate_color(triangle, index, delta)
    return triangle


def generate_random_triangle(canvas_size):
    """Generate a random triangle with clamped RGBA values."""
    width, height = canvas_size
    triangle = {
        "x1": random.randint(0, width),
        "y1": random.randint(0, height),
        "x2": random.randint(0, width),
        "y2": random.randint(0, height),
        "x3": random.randint(0, width),
        "y3": random.randint(0, height),
        "color": (
            random.randint(0, 180),  # R
            random.randint(0, 180),  # G
            random.randint(0, 180),  # B
            random.randint(30, 150)  # A
        )
    }
    return triangle
