# Fitness functions for both exercises
import numpy as np
from PIL import Image


def compute_mse(image1, image2):
    """
    Compute Mean Squared Error between two numpy arrays.
    Assumes image1 and image2 are of the same shape.
    """

    return np.mean((image1 - image2) ** 2)


def compute_triangle_fitness(individual, target_image):
    """
    Evaluate fitness for a triangle-based individual.

    Parameters:
    - individual: A TriangleIndividual with a rendered output.
    - target_image: A PIL Image of the target.

    Steps:
    1. Render the individual to get the generated image.
    2. Convert both the generated image and the target image to grayscale.
    3. Calculate MSE between the two images.
    4. Return a fitness value that increases as the error decreases.
    """

    rendered_image = individual.render()
    # Convert images to grayscale
    rendered_gray = np.array(rendered_image.convert('L'))
    target_gray = np.array(target_image.convert('L'))

    mse_value = compute_mse(rendered_gray, target_gray)
    fitness = 1 / (1 + mse_value)
    return fitness
