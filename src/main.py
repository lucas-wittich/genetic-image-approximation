import json
from PIL import Image
from ga_engine import GAEngine
import os


def main():
    # Load configuration from config.json
    with open('../configs/config.json', "r") as f:
        config = json.load(f)

    # Load the target image and convert it to RGBA
    target_image = Image.open(config["target_image"]).convert("RGBA")

    # Get canvas size as a tuple (or simply use target_image.size)
    canvas_size = target_image.size

    # Extract GA parameters from the config file
    num_triangles = config["num_triangles"]
    population_size = config["population_size"]
    num_generations = config["num_generations"]
    mutation_rate = config["mutation_rate"]
    crossover_rate = config["crossover_rate"]
    num_mutated_genes = config["num_mutated_genes"]
    delta = config["delta"]
    mutation_strategy = config["mutation_strategy"]
    selection_method = config["selection_method"]
    selection_params = config.get("selection_params", {})

    # Extract termination parameters
    termination_params = config.get("termination", {})
    window_size = termination_params.get("window_size", 10)
    stagnation_threshold = termination_params.get("stagnation_threshold", 0.001)
    diversity_threshold = termination_params.get("diversity_threshold", 0.1)

    # Initialize the GA engine with all parameters from the config
    engine = GAEngine(
        target_image=target_image,
        canvas_size=canvas_size,
        num_triangles=num_triangles,
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        num_mutated_genes=num_mutated_genes,
        selection_method=selection_method,
        selection_params=selection_params,
        delta=delta,
        mutation_strategy=mutation_strategy,
        window_size=window_size,
        stagnation_threshold=stagnation_threshold,
        diversity_threshold=diversity_threshold
    )

    # Run the evolution process
    best_individual, fitness_history = engine.evolve()

    # Render the phenotype (output image) of the best individual
    output_image = best_individual.render()

    # Create the output directory if it doesn't exist
    output_directory = "../data/outputs"
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, "best_output.png")
    output_image.save(output_path)

    # Print summary information
    print(f"Evolution complete.\nBest Fitness: {best_individual.fitness}")
    print(f"Output image saved at: {output_path}")


if __name__ == "__main__":
    main()
