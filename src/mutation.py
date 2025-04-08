# Mutation strategies (bit-level, multi-gene, etc.)
import random

width_keys = ['x1', 'x2', 'x3']
height_keys = ['y1', 'y2', 'y3']
coord_keys = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3']
color_keys = [("color", i) for i in range(4)]
all_genes = color_keys + coord_keys


def delta_mutation(gene_value, delta):
    adjustment = random.uniform(-delta, delta)
    return gene_value + adjustment


def single_gene_mutation(triangle, mutation_rate, delta, canvas_size):
    if random.random() < mutation_rate:
        chosen_gene = random.choice(all_genes)
        width, height = canvas_size
        if isinstance(chosen_gene, str):
            if chosen_gene.startswith('x'):
                mutated_value = delta_mutation(triangle[chosen_gene], delta)
                triangle[chosen_gene] = int(max(0, min(mutated_value, width)))
            else:
                mutated_value = delta_mutation(triangle[chosen_gene], delta)
                triangle[chosen_gene] = int(max(0, min(mutated_value, height)))
        else:
            key, index = chosen_gene
            color = list(triangle[key])
            new_value = delta_mutation(color[index], delta)
            color[index] = int(max(0, min(new_value, 255)))
            triangle[key] = tuple(color)
    return triangle


def limited_multi_gene_mutation(triangle, delta, canvas_size, mutation_rate, num_mutated_genes=None):
    if random.random() < mutation_rate:
        M = len(all_genes)
        if num_mutated_genes is None:
            num_mutated_genes = random.randint(1, M)
        selected_genes = random.sample(all_genes, num_mutated_genes)
        width, height = canvas_size
        for gene in selected_genes:
            if isinstance(gene, str):
                if gene.startswith("x"):
                    mutated_value = delta_mutation(triangle[gene], delta)
                    triangle[gene] = int(max(0, min(mutated_value, width)))
                elif gene.startswith("y"):
                    mutated_value = delta_mutation(triangle[gene], delta)
                    triangle[gene] = int(max(0, min(mutated_value, height)))
            else:
                key, i = gene
                color = list(triangle[key])
                mutated_value = delta_mutation(color[i], delta)
                color[i] = int(max(0, min(mutated_value, 255)))
                triangle[key] = tuple(color)

    return triangle


def uniform_multi_gene_mutation(triangle, mutation_rate, delta, canvas_size):
    width, height = canvas_size
    for key in coord_keys:
        if random.random() < mutation_rate:
            if key.startswith("x"):
                new_value = delta_mutation(triangle[key], delta)
                triangle[key] = int(max(0, min(new_value, width)))
            elif key.startswith("y"):
                new_value = delta_mutation(triangle[key], delta)
                triangle[key] = int(max(0, min(new_value, height)))
    for gene in color_keys:
        key, index = gene
        if random.random() < mutation_rate:
            color = list(triangle[key])
            new_value = delta_mutation(color[index], delta)
            color[index] = int(max(0, min(new_value, 255)))
            triangle[key] = tuple(color)
    return triangle


def complete_mutation(triangle, mutation_rate, delta, canvas_size):
    if random.random() < mutation_rate:
        coord_genes = ["x1", "y1", "x2", "y2", "x3", "y3"]
        color_genes = [("color", i) for i in range(4)]
        width, height = canvas_size
        for key in coord_genes:
            if key.startswith("x"):
                new_value = delta_mutation(triangle[key], delta)
                triangle[key] = int(max(0, min(new_value, width)))
            elif key.startswith("y"):
                new_value = delta_mutation(triangle[key], delta)
                triangle[key] = int(max(0, min(new_value, height)))
        for gene in color_genes:
            key, index = gene
            color = list(triangle[key])
            new_value = delta_mutation(color[index], delta)
            color[index] = int(max(0, min(new_value, 255)))
            triangle[key] = tuple(color)
    return triangle


def generate_random_trianle(canvas_size):
    width, height = canvas_size
    triangle = {
        "x1": random.randint(0, width),
        "y1": random.randint(0, height),
        "x2": random.randint(0, width),
        "y2": random.randint(0, height),
        "x3": random.randint(0, width),
        "y3": random.randint(0, height),
        "color": (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(30, 180)
        )
    }
    return triangle
