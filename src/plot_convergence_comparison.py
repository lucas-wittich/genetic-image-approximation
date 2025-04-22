import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def plot_comparison(csv_dir, method_col, label=None):
    files = sorted(glob.glob(f"{csv_dir}/*_comparison.csv"))
    plt.figure(figsize=(12, 8))

    for file in files:
        df = pd.read_csv(file)
        if method_col not in df.columns:
            continue
        for method, group in df.groupby(method_col):
            grouped = group.groupby("generation")["inv_best_fitness"].mean().reset_index()
            plt.plot(grouped["generation"], grouped["inv_best_fitness"], label=f"{label or method} - {method}")

    plt.title("Fitness Convergence Comparison")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (↑)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{csv_dir}/convergence_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_comparison("../data/results/mutation", "mutation_strategy")
    plot_comparison("../data/results/selection", "selection_method")
    plot_comparison("../data/results/crossover", "crossover_method")
