from mutation import complete_mutation, single_gene_mutation, limited_multi_gene_mutation, uniform_multi_gene_mutation
import random
import copy
from PIL import Image, ImageDraw


class TriangleIndividual:
    def __init__(self, triangles, canvas_size, fitness=None):
        """
        Initialize a TriangleIndividual.

        Parameters:
        - triangles: a list of dictionaries, each representing a triangle with:
            - x1, y1, x2, y2, x3, y3: coordinates of the three points
            - color: a tuple (R, G, B, A) representing the color and transparency
        - canvas_size: a tuple (width, height) for the drawing canvas
        """
        self.triangles = triangles
        self.canvas_size = canvas_size
        self.fitness = fitness

    @classmethod
    def random_initialize(cls, num_triangles, canvas_size):
        """
        Create a TriangleIndividual with a random genotype.

        Parameters:
        - num_triangles: number of triangles in the genotype
        - canvas_size: a tuple (width, height) for the drawing canvas

        Returns:
        - A TriangleIndividual instance with randomly generated triangle data.
        """
        width, height = canvas_size
        triangles = []
        for _ in range(num_triangles):
            triangle = {
                "x1": random.randint(0, width),
                "y1": random.randint(0, height),
                "x2": random.randint(0, width),
                "y2": random.randint(0, height),
                "x3": random.randint(0, width),
                "y3": random.randint(0, height),
                "color": (
                    random.randint(0, 255),  # Red
                    random.randint(0, 255),  # Green
                    random.randint(0, 255),  # Blue
                    random.randint(20, 150)  # Alpha (translucency)
                )
            }
            triangles.append(triangle)
        return cls(triangles, canvas_size)

    def clone(self):
        """
        Create a deep copy of this individual.

        This is important to avoid side effects when modifying individuals
        during mutation or crossover.
        """
        return TriangleIndividual(copy.deepcopy(self.triangles), self.canvas_size, self.fitness)

    def mutate(self, mutation_rate=0.01, delta=10, mutation_strategy="uniform", num_mutated_genes=None):
        """
        Apply mutation to the individual's triangles based on the specified strategy.

        Parameters:
        - mutation_rate: Probability of mutation for each triangle.
        - delta: Maximum change applied during mutation.
        - mutation_strategy: Strategy to use for mutation ('single', 'limited', 'complete', 'uniform').
        - num_mutated_genes: Number of genes to mutate (used in 'limited' strategy).
        """
        new_triangles = []
        for triangle in self.triangles:
            if mutation_strategy == "single":
                mutated_triangle = single_gene_mutation(triangle, mutation_rate, delta, self.canvas_size)
            elif mutation_strategy == "limited":
                mutated_triangle = limited_multi_gene_mutation(
                    triangle, delta, self.canvas_size, mutation_rate, num_mutated_genes)
            elif mutation_strategy == "complete":
                mutated_triangle = complete_mutation(triangle, mutation_rate, delta, self.canvas_size)
            else:
                mutated_triangle = uniform_multi_gene_mutation(triangle, mutation_rate, delta, self.canvas_size)
            new_triangles.append(mutated_triangle)
        self.triangles = new_triangles

    def render(self):
        """
        Render the phenotype with proper alpha compositing using Pillow.
        Returns:
            A PIL Image object representing the rendered individual.
        """
        # Create a white RGBA base image
        base = Image.new("RGBA", self.canvas_size, (255, 255, 255, 255))

        for triangle in self.triangles:
            # Create a transparent layer
            layer = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer, "RGBA")

            # Define triangle points
            points = [
                (triangle["x1"], triangle["y1"]),
                (triangle["x2"], triangle["y2"]),
                (triangle["x3"], triangle["y3"]),
            ]

            # Draw the triangle on the transparent layer
            draw.polygon(points, fill=triangle["color"])

            # Composite the layer onto the base image
            base = Image.alpha_composite(base, layer)

        return base
