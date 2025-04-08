# Individual representation (ASCII or triangle-based)
import random
import copy
from PIL import Image, ImageDraw


class TriangleIndividual:
    def __init__(self, triangles, canvas_size):
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
        self.fitness = None

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
                    random.randint(30, 180)  # (translucency)
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

        return TriangleIndividual(copy.deepcopy(self.triangles), self.canvas_size)

    def mutate(self, mutation_rate=0.01, delta=10):
        """
        Apply mutations to the individual's genotype.

        Each triangle has a chance to have one of its point coordinates or one
        of its color components adjusted by a random delta.

        Parameters:
        - mutation_rate: probability of mutating each attribute
        - delta: maximum change applied during mutation
        """

        width, height = self.canvas_size
        for triangle in self.triangles:
            # Mutate a coordinate of a randomly chosen vertex
            if random.random() < mutation_rate:
                point = random.choice(["x1", "y1", "x2", "y2", "x3", "y3"])
                triangle[point] += random.randint(-delta, delta)
                # Ensure the coordinate stays within canvas bounds
                triangle[point] = max(0, min(triangle[point], width if 'x' in point else height))

            # Mutate one of the color components
            if random.random() < mutation_rate:
                c_idx = random.randint(0, 3)
                color = list(triangle["color"])
                color[c_idx] += random.randint(-delta, delta)
                color[c_idx] = max(0, min(color[c_idx], 255))
                triangle["color"] = tuple(color)

    def render(self):
        """
        Render the phenotype from the genotype.

        Returns:
        - A PIL Image showing all triangles drawn on a blank white canvas.
        """

        img = Image.new("RGBA", self.canvas_size, (255, 255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        for triangle in self.triangles:
            points = [
                (triangle["x1"], triangle["y1"]),
                (triangle["x2"], triangle["y2"]),
                (triangle["x3"], triangle["y3"]),
            ]
            draw.polygon(points, fill=triangle["color"])
        return img
