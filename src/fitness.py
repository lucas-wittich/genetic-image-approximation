
import numpy as np
from PIL import Image


def normalize_image(img):
    """Convert PIL RGBA image to a normalized numpy array (0–1 range)."""
    return np.asarray(img).astype(np.float32) / 255.0


def premultiply_rgb(img):
    """Apply premultiplied alpha to RGB channels (perceptual blending)."""
    rgb = img[..., :3]
    alpha = img[..., 3:]
    return rgb * alpha


def compute_alpha_weight(img1, img2):
    """Compute per-pixel alpha weight for RGB loss, based on transparency."""
    a1 = img1[..., 3]
    a2 = img2[..., 3]
    return np.clip((a1 + a2) / 2.0, 0.001, 1.0)  # avoid division by zero


def triangle_area(triangle):
    """Calculate area of a triangle given its vertex coordinates."""
    x1, y1 = triangle["x1"], triangle["y1"]
    x2, y2 = triangle["x2"], triangle["y2"]
    x3, y3 = triangle["x3"], triangle["y3"]
    return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0)


def compute_triangle_fitness(individual, target_img):
    """
    Compute fitness based on alpha-aware RGB difference + alpha loss + area penalty.

    Lower fitness is better.
    """
    canvas = individual.render()
    canvas_np = normalize_image(canvas)
    target_np = normalize_image(target_img)

    # Premultiplied RGB loss (alpha-weighted)
    rgb1 = premultiply_rgb(canvas_np)
    rgb2 = premultiply_rgb(target_np)
    alpha_weight = compute_alpha_weight(canvas_np, target_np)
    mse_rgb = np.mean(alpha_weight[..., None] * (rgb1 - rgb2) ** 2)

    # Alpha loss (unweighted MSE on alpha channel)
    mse_alpha = np.mean((canvas_np[..., 3] - target_np[..., 3]) ** 2)

    # Triangle area penalty (regularization)
    canvas_area = canvas_np.shape[0] * canvas_np.shape[1]
    avg_area = np.mean([triangle_area(t) for t in individual.triangles])
    area_penalty_weight = 0.2
    area_penalty = (avg_area / canvas_area) * area_penalty_weight
    error = mse_rgb + 0.25 * mse_alpha + area_penalty
    return 1 / (1 + error)
