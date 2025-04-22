import numpy as np

# Module‑level caches
_TARGET_NORM = None       # normalized RGBA target array
_TARGET_PRE = None        # premultiplied RGB target array
_CANVAS_AREA = None       # total pixel count


def init_target(target_img):
    """
    Initialize and cache the target image arrays.
    """
    global _TARGET_NORM, _TARGET_PRE, _CANVAS_AREA
    arr = np.asarray(target_img.convert("RGBA")).astype(np.float32) / 255.0
    _TARGET_NORM = arr
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    _TARGET_PRE = rgb * alpha
    _CANVAS_AREA = arr.shape[0] * arr.shape[1]


def compute_triangle_fitness(individual, _ignored=None):
    """
    Compute fitness of an individual against the cached target.

    Returns a fitness in [0,1], higher is better.
    """
    # 1) Render and normalize canvas
    canvas = individual.render()
    canvas_arr = np.asarray(canvas).astype(np.float32) / 255.0

    # 2) Premultiply canvas RGB
    rgb_c = canvas_arr[..., :3]
    alpha_c = canvas_arr[..., 3:4]
    canvas_pre = rgb_c * alpha_c

    # 3) MSE on premultiplied RGB channels
    mse_rgb = np.mean((canvas_pre - _TARGET_PRE) ** 2)

    # 4) Alpha channel MSE
    mse_alpha = np.mean((canvas_arr[..., 3] - _TARGET_NORM[..., 3]) ** 2)

    # 5) Triangle area penalty
    def triangle_area(t):
        x1, y1 = t['x1'], t['y1']
        x2, y2 = t['x2'], t['y2']
        x3, y3 = t['x3'], t['y3']
        return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0)

    avg_area = np.mean([triangle_area(t) for t in individual.triangles])
    area_penalty = (avg_area / _CANVAS_AREA) * 0.2

    # 6) Combine errors and convert to fitness
    raw_error = mse_rgb + 0.25 * mse_alpha + area_penalty
    return 1.0 / (1.0 + raw_error)
