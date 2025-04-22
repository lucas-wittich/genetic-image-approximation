# Image processing, helper functions, logging, etc.
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image(path):
    return Image.open(path).convert("RGB")


def resize_image(image, width, height):
    return image.resize((width, height), Image.LANCZOS)  # Use high-quality resampling


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def plot_fitness_metrics(fitness_history, error_history=None, output_dir=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, label="Fitness (↑)")
    if error_history:
        plt.plot(error_history, label="Error (↓)")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title("Fitness and Error over Generations")
    plt.legend()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "fitness_plot.png"))
    else:
        plt.show()


def plot_fitness_metrics(min_hist, avg_hist, max_hist, diversity_hist, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    generations = range(len(min_hist))  # all metrics have the same length

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Fitness curves (min, avg, max)
    axs[0].plot(generations, min_hist, label="Min Fit", color="orange")
    axs[0].plot(generations, avg_hist, label="Avg Fit", color="blue")
    axs[0].plot(generations, max_hist, label="Max Fit", color="green")
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("Fitness")
    axs[0].set_title("Min/Avg/Max Fitness Evolution")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Diversity
    axs[1].plot(generations, diversity_hist, label="Diversity", color="red")
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Diversity (std dev of fitness)")
    axs[1].set_title("Population Diversity")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
