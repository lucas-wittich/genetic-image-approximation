
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from ga_engine import GAEngine


def save_gif(frames, path, duration=300):
    """Save a sequence of PIL images as an animated GIF."""
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)


def main():
    with open('../configs/config_high_quality.json', "r") as f:
        config = json.load(f)

    target_image = Image.open(config["target_image"]).convert("RGBA")
    canvas_size = target_image.size
    runs = config.get("runs_per_config", 5)

    all_stats = []
    all_snapshots = []
    best_overall = None
    best_final_fitness = float('inf')

    for run in range(runs):
        print(f"\n=== Run {run + 1}/{runs} ===")
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
            selection_params=config.get("selection_params", {}),
            mutation_strategy=config["mutation_strategy"],
            termination_params=config.get("termination", {})
        )
        best_individual, stats = engine.evolve()
        final_fitness = best_individual.fitness
        all_stats.append(stats)

        if final_fitness < best_final_fitness:
            best_final_fitness = final_fitness
            best_overall = best_individual
            all_snapshots = stats["snapshots"]

    # Save best output
    output_dir = "../data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    best_overall.render().save(os.path.join(output_dir, "best_output.png"))

    # Save fitness stats (averaged across runs)
    generations = len(all_stats[0]["best_fitness"])
    df = pd.DataFrame({
        "generation": list(range(generations)),
        "best_fitness": [pd.Series(run["best_fitness"]) for run in all_stats],
        "avg_fitness": [pd.Series(run["avg_fitness"]) for run in all_stats],
        "diversity": [pd.Series(run["diversity"]) for run in all_stats]
    }).explode(["best_fitness", "avg_fitness", "diversity"]).astype({
        "generation": int,
        "best_fitness": float,
        "avg_fitness": float,
        "diversity": float
    })

    df.to_csv(os.path.join(output_dir, "fitness_history.csv"), index=False)

    # Plot averaged fitness evolution
    plt.figure(figsize=(10, 6))
    df_grouped = df.groupby("generation").agg({
        "best_fitness": ["mean", "std"],
        "avg_fitness": ["mean", "std"]
    })
    generations = df_grouped.index
    best_mean = df_grouped[("best_fitness", "mean")]
    best_std = df_grouped[("best_fitness", "std")]
    avg_mean = df_grouped[("avg_fitness", "mean")]
    avg_std = df_grouped[("avg_fitness", "std")]

    plt.plot(generations, best_mean, label="Best Fitness", linewidth=2)
    plt.fill_between(generations, best_mean - best_std, best_mean + best_std, alpha=0.2)
    plt.plot(generations, avg_mean, linestyle="--", label="Avg Fitness")
    plt.fill_between(generations, avg_mean - avg_std, avg_mean + avg_std, alpha=0.1)
    plt.title("Fitness Evolution Across Runs")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fitness_plot.png"))
    plt.close()

    # Save animated evolution GIF
    gif_path = os.path.join(output_dir, "evolution.gif")
    save_gif(all_snapshots, gif_path, duration=300)

    print(f"\nEvolution complete. Best fitness: {best_final_fitness:.6f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
