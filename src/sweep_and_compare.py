
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import resize_image
from ga_engine import GAEngine


CONFIG_TEMPLATE = {
    "canvas_size": [64, 64],
    "num_triangles": 50,
    "population_size": 40,
    "num_generations": 150,
    "mutation_rate": 0.2,
    "crossover_rate": 0.6,
    "num_mutated_genes": 3,
    "runs_per_config": 3,
    "mutation_strategy": "limited",
    "crossover_method": "uniform",
    "target_image_path": "../data/inputs/input_image.jpg",
    "termination": {
        "window_size": 10,
        "stagnation_threshold": 0.001,
        "diversity_threshold": 0.001
    }
}

experiments = []

for temp in [0.1, 0.5, 1.0, 2.0]:
    config = CONFIG_TEMPLATE.copy()
    config.update({
        "selection_method": "boltzmann",
        "selection_params": {"temperature": temp}
    })
    experiments.append(("boltzmann", f"temp{temp}", config))

for size in [3, 5, 7]:
    config = CONFIG_TEMPLATE.copy()
    config.update({
        "selection_method": "tournament",
        "selection_params": {"tournament_size": size, "deterministic": True}
    })
    experiments.append(("tournament", f"size{size}", config))

for delta in [5, 10, 20]:
    config = CONFIG_TEMPLATE.copy()
    config.update({
        "selection_method": "tournament",
        "selection_params": {"tournament_size": 5, "deterministic": True},
        "delta": delta
    })
    experiments.append(("delta", f"delta{delta}", config))

os.makedirs("sweep_results", exist_ok=True)


def run_and_plot(category, tag, config, idx):
    target_image = Image.open(config["target_image_path"]).convert("RGBA")
    canvas_size = config.get("canvas_size", target_image.size)
    target_image = resize_image(target_image, canvas_size[0], canvas_size[1])
    all_records = []

    for run in range(config["runs_per_config"]):
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
            generation_approach=config.get("generation_approach", "traditional"),
            elitism_rate=config.get("elitism_rate", 0.2),
            delta=config.get("delta", 10),
            young_bias_ratio=config.get("young_bias_ratio", 1.0)
        )
        best_ind, stats = engine.evolve()
        for gen, (b, a, d) in enumerate(zip(stats["best_fitness"], stats["avg_fitness"], stats["diversity"])):
            all_records.append({
                "generation": gen,
                "best_fitness": b,
                "avg_fitness": a,
                "diversity": d,
                "inv_best_fitness": 1.0 / (b + 1e-6),
                "normalized_fitness": 1 - b / stats["best_fitness"][0],
                "label": f"{category}_{tag}_{idx}",
                "run": run
            })

    df = pd.DataFrame(all_records)
    df.to_csv(f"sweep_results/{category}_{tag}_{idx}.csv", index=False)
    print(f"Saved: sweep_results/{category}_{tag}_{idx}.csv")
    return df


if __name__ == "__main__":
    all_dfs = []
    for idx, (category, tag, config) in enumerate(experiments):
        df = run_and_plot(category, tag, config, idx)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv("sweep_results/all_sweeps_combined.csv", index=False)

        plt.figure(figsize=(14, 8))
        for label, group in full_df.groupby("label"):
            grouped = group.groupby("generation")["inv_best_fitness"].mean()
            plt.plot(grouped.index, grouped.values, label=label)
        plt.title("Sweep Comparison - Rising Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Inverted Fitness (↑ = better)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("sweep_results/sweep_convergence.png", dpi=300)
        plt.show()

        leaderboard = full_df.groupby("label").agg({
            "best_fitness": "min",
            "normalized_fitness": "max",
            "diversity": "last"
        }).sort_values(by="best_fitness")
        print("\n Final Leaderboard:\n", leaderboard)
