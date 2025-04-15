import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from utils import resize_image
from ga_engine import GAEngine
import numpy as np

CONFIG_TEMPLATE = {
    "canvas_size": [64, 64],
    "num_triangles": 50,
    "population_size": 40,
    "num_generations": 150,
    "mutation_rate": 0.2,
    "crossover_rate": 0.6,
    "num_mutated_genes": 3,
    "runs_per_config": 5,
    "mutation_strategy": "limited",
    "crossover_method": "uniform",
    "selection_method": "tournament",
    "selection_params": {"tournament_size": 5, "deterministic": True},
    "target_image_path": "../data/inputs/input_image.jpg",
    "termination": {
        "window_size": 10,
        "stagnation_threshold": 0.001,
        "diversity_threshold": 0.001
    }
}

strategies = [
    {"generation_approach": "traditional", "elitism_rate": 0.2, "young_bias_ratio": 1.0},
    {"generation_approach": "young_bias", "elitism_rate": 0.1, "young_bias_ratio": 0.7}
]


def run_strategy(name, config):
    os.makedirs("sweep_results", exist_ok=True)
    target_image = Image.open(config["target_image_path"]).convert("RGBA")
    canvas_size = config.get("canvas_size", target_image.size)
    target_image = resize_image(target_image, canvas_size[0], canvas_size[1])

    all_records = []
    for run in range(config["runs_per_config"]):
        print(f"▶ Running strategy: {name} | Run {run+1}")
        engine = GAEngine(
            target_image=target_image,
            canvas_size=config["canvas_size"],
            num_triangles=config["num_triangles"],
            population_size=config["population_size"],
            num_generations=config["num_generations"],
            mutation_rate=config["mutation_rate"],
            crossover_rate=config["crossover_rate"],
            num_mutated_genes=config["num_mutated_genes"],
            selection_method=config["selection_method"],
            selection_params=config["selection_params"],
            mutation_strategy=config["mutation_strategy"],
            termination_params=config["termination"],
            generation_approach=config["generation_approach"],
            elitism_rate=config["elitism_rate"],
            delta=config.get("delta", 10),
            young_bias_ratio=config["young_bias_ratio"]
        )
        best_ind, stats = engine.evolve()

        for gen, (b, a, d, n) in enumerate(zip(stats["best_fitness"],
                                               stats["avg_fitness"],
                                               stats["diversity"],
                                               stats["normalized_fitness"])):
            all_records.append({
                "generation": gen,
                "best_fitness": b,
                "avg_fitness": a,
                "diversity": d,
                "normalized_fitness": n,
                "inv_best_fitness": 1.0 / (b + 1e-6),
                "label": name,
                "run": run
            })

    df = pd.DataFrame(all_records)
    df.to_csv(f"sweep_results/{name}.csv", index=False)
    return df


if __name__ == "__main__":
    all_dfs = []
    for strategy in strategies:
        config = CONFIG_TEMPLATE.copy()
        config.update(strategy)
        df = run_strategy(strategy["generation_approach"], config)
        all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv("sweep_results/generation_comparison.csv", index=False)

        plt.figure(figsize=(12, 8))
        for label, group in full_df.groupby("label"):
            grouped = group.groupby("generation")["normalized_fitness"].agg(["mean", "std"]).reset_index()

            plt.plot(grouped["generation"], grouped["mean"], label=label)
            plt.fill_between(grouped["generation"],
                             grouped["mean"] - grouped["std"],
                             grouped["mean"] + grouped["std"],
                             alpha=0.2)
        plt.title("Generation Strategy Comparison (Young-Bias vs Traditional)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (↑ = better)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("sweep_results/generation_comparison.png", dpi=300)
        plt.show()

        leaderboard = full_df.groupby("label").agg({
            "best_fitness": "min",
            "normalized_fitness": "max",
            "diversity": "last"
        }).sort_values(by="best_fitness")
        print("\nFinal Leaderboard:")
        print(leaderboard)
