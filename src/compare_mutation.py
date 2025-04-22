import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import resize_image
from ga_engine import GAEngine
from PIL import Image


def run_mutation_experiment(config):
    mutation_strategies = ["single", "limited", "uniform", "complete"]
    target_path = config.get("target_image_path", config.get("target_image"))
    target_image = Image.open(target_path).convert("RGBA")
    canvas_size = config.get("canvas_size", target_image.size)
    target_image = resize_image(target_image, canvas_size[0], canvas_size[1])
    os.makedirs("../data/results/mutation", exist_ok=True)
    all_records = []

    for strategy in mutation_strategies:
        print(f"Running mutation strategy: {strategy}")
        for run_id in range(config["runs_per_config"]):
            print(f"  Run {run_id + 1}/{config['runs_per_config']}")
            engine = GAEngine(
                target_image=target_image,
                canvas_size=canvas_size,
                num_triangles=config["num_triangles"],
                population_size=config["population_size"],
                num_generations=config["num_generations"],
                mutation_rate=config["mutation_rate"],
                crossover_rate=config["crossover_rate"],
                num_mutated_genes=config["num_mutated_genes"],
                selection_method=config["selection_method"],
                selection_params=config["selection_params"],
                mutation_strategy=strategy,
                termination_params=config.get("termination", {})
            )

            best_individual, stats = engine.evolve()

            for generation, best_fitness in enumerate(stats["best_fitness"]):
                avg_fitness = stats["avg_fitness"][generation]
                diversity = stats["diversity"][generation]
                if generation == 0:
                    initial_fitness = best_fitness
                    fitness_delta = 0
                    norm_fitness = 0
                else:
                    fitness_delta = prev_best - best_fitness
                    norm_fitness = 1 - (best_fitness / initial_fitness)
                prev_best = best_fitness

                all_records.append({
                    "mutation_strategy": strategy,
                    "run": run_id,
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "avg_fitness": avg_fitness,
                    "diversity": diversity,
                    "fitness_delta": fitness_delta,
                    "normalized_fitness": norm_fitness
                })

    df = pd.DataFrame(all_records)
    df["inv_best_fitness"] = 1.0 / (df["best_fitness"] + 1e-6)
    df["inv_avg_fitness"] = 1.0 / (df["avg_fitness"] + 1e-6)
    df.to_csv("../data/results/mutation/mutation_comparison.csv", index=False)

    plt.figure(figsize=(12, 8))
    for strategy, group in df.groupby("mutation_strategy"):
        grouped = group.groupby("generation").agg({
            "inv_best_fitness": ["mean", "std"],
            "inv_avg_fitness": ["mean", "std"]
        }).reset_index()
        generations = grouped["generation"]
        best_mean = grouped[("inv_best_fitness", "mean")]
        best_std = grouped[("inv_best_fitness", "std")]
        avg_mean = grouped[("inv_avg_fitness", "mean")]
        avg_std = grouped[("inv_avg_fitness", "std")]

        plt.plot(generations, best_mean, label=f"{strategy} (best↑)", linewidth=2)
        plt.fill_between(generations, best_mean - best_std, best_mean + best_std, alpha=0.2)
        plt.plot(generations, avg_mean, linestyle="--", label=f"{strategy} (avg↑)")
        plt.fill_between(generations, avg_mean - avg_std, avg_mean + avg_std, alpha=0.1)

    plt.title("Mutation Strategy Comparison")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (↑ = better)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../data/results/mutation/mutation_comparison.png", dpi=300)
    plt.savefig("../data/results/mutation/mutation_comparison.svg")
    plt.show()


def load_config(defaults):
    parser = argparse.ArgumentParser(description="Compare Mutation Strategies in Genetic Algorithm")
    parser.add_argument("--config", type=str, help="Path to config JSON file", default=None)
    args = parser.parse_args()

    config = defaults.copy()
    if args.config and os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    return config


default_config = {
    "canvas_size": [64, 64],
    "num_triangles": 50,
    "population_size": 30,
    "num_generations": 100,
    "mutation_rate": 0.2,
    "crossover_rate": 0.5,
    "delta": 10,
    "num_mutated_genes": 3,
    "selection_method": "tournament",
    "selection_params": {"tournament_size": 5, "deterministic": True},
    "runs_per_config": 5,
    "target_image_path": "../data/inputs/input_image.jpg"
}

if __name__ == "__main__":
    config = load_config(default_config)
    run_mutation_experiment(config)
