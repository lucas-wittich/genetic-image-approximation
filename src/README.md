# TP2 - Genetic Algorithms

This project implements a Genetic Algorithm (GA) engine applied to two problems:

1. ASCII art generation from square images.
2. Image approximation using translucent triangles.

---

## 🧠 Objectives

### ✅ Build a modular Genetic Algorithm engine
- [ ] Separate components for selection, crossover, mutation, and fitness evaluation.
- [ ] Configurable strategies and parameters via a configuration file.

### ✅ Exercise 1 - ASCII Art
- [ ] Load a square image.
- [ ] Represent it using NxN ASCII characters.
- [ ] Define a fitness function to evaluate ASCII image quality.
- [ ] Output: ASCII image + fitness score.

### ✅ Exercise 2 - Translucent Triangles
- [ ] Input: image and number of triangles.
- [ ] Individuals are composed of triangle sets (position, color, opacity).
- [ ] Define a fitness function based on pixel similarity.
- [ ] Output: generated image + list of triangle data.

### ✅ Genetic Algorithm Features
- [ ] Implement multiple parent selection methods: Roulette, Ranking, Tournament.
- [ ] Support different crossover types: One-point, Two-point, Uniform, Annular.
- [ ] Implement mutation variations: Bit flip, delta change, multi-gene, full.
- [ ] Define flexible stopping conditions: generation count, fitness stagnation, etc.
- [ ] Avoid premature convergence and ensure diversity.

---

## ⚙️ Configuration

Parameters are defined in `config.yml`, such as:
- Problem type (`ascii` or `triangles`)
- Population size
- Crossover probability
- Mutation probability
- Selection/crossover/mutation method
- Number of generations
- Termination criteria
- Input image path
- Number of triangles (for triangle mode only)


---

## 3. Detailed File Contents and Methods

### 3.1. `main.py`
- **Purpose:** Acts as the program entry point.
- **Responsibilities:**
  - **Load configuration:** Read `config.yml` for GA parameters (population size, mutation rate, etc.).
  - **Load target image:** Use a utility function (from `utils.py`) to load the image from `data/input_image.jpg`.
  - **Initialize GAEngine:** Create an instance of `GAEngine` using the loaded parameters.
  - **Run the evolution:** Call the engine’s `evolve()` method.
  - **Output results:** Save the best individual’s rendered image in `data/outputs/` and log performance metrics.
- **Methods to Include:**
  - `def main():` — main function to tie together configuration loading, engine initialization, evolution, and output.

---

### 3.2. `triangle_individual.py`
- **Purpose:** Defines the `TriangleIndividual` class representing a candidate solution.
- **Responsibilities:**
  - **Genotype:** Maintain a list of triangles (each defined by three points and a color in RGBA).
  - **Phenotype:** Render the genotype into an image.
  - **Genetic operations:** Provide methods for random initialization, cloning, and mutation.
- **Required Methods:**
  - `__init__(self, triangles, canvas_size)`: Initialize with a given list of triangle dictionaries and canvas size.
  - `@classmethod def random_initialize(cls, num_triangles, canvas_size)`: Return a new instance with randomly generated triangles.
  - `def clone(self)`: Return a deep copy of the instance (ensuring no shared references).
  - `def mutate(self, mutation_rate=0.01, delta=10)`: Randomly adjust triangle coordinates and color components.
  - `def render(self)`: Draw triangles onto a blank canvas using Pillow and return a PIL.Image.

---

### 3.3. `ga_engine.py`
- **Purpose:** Implements the core GA loop using `TriangleIndividual`.
- **Responsibilities:**
  - **Population Initialization:** Generate the initial population of individuals.
  - **Fitness Evaluation:** Compute fitness for each individual using functions from `fitness.py`.
  - **Parent Selection:** Use a configurable selection method (imported from `selection.py`).
  - **Crossover and Mutation:** Apply genetic operators to produce offspring.
  - **Termination:** Run until a termination condition is met (see `termination.py`).
- **Required Methods:**
  - `__init__(self, target_image, canvas_size, num_triangles, population_size, num_generations, mutation_rate, crossover_rate, selection_method="tournament", selection_params=None)`: Set all GA parameters and choose the selection method.
  - `def initialize_population(self)`: Create the initial population using `TriangleIndividual.random_initialize()`.
  - `def evaluate_fitness(self)`: Loop over the population and update each individual’s `fitness` by calling `compute_triangle_fitness()`.
  - `def select_parents(self)`: Call the configured selection function (from `selection.py`) to choose parents.
  - `def crossover(self, parent1, parent2)`: Perform one-point crossover on the parents’ genotypes and return two new offspring.
  - `def mutate(self, individual)`: Invoke the individual's mutation method.
  - `def evolve(self)`: Run the main GA loop:
    - Loop through generations.
    - Perform selection, crossover, mutation.
    - Evaluate fitness and update the best individual.
    - Optionally check for termination using methods from `termination.py`.
    - Return the best individual.

---

### 3.4. `selection.py`
- **Purpose:** Contains all selection methods and a helper to configure them.
- **Responsibilities:** Provide multiple selection strategies that can be chosen via a configuration parameter.
- **Required Functions:**
  - `def roulette_selection(population, k)`: Implement fitness-proportionate selection.
  FUNCTION roulette_selection(population, k):
    total_fitness = SUM( f for each individual with fitness f in population )
    selected = EMPTY LIST
    FOR iteration FROM 1 TO k:
        r = RANDOM_NUMBER between 0 and total_fitness
        cumulative = 0
        FOR each individual in population:
            cumulative += individual.fitness
            IF cumulative >= r THEN
                ADD individual.clone() TO selected
                BREAK
            END IF
        END FOR
    END FOR
    RETURN selected
END FUNCTION

  - `def tournament_selection(population, k, tournament_size=5, deterministic=True)`: Implement tournament selection.
  FUNCTION tournament_selection(population, k, tournament_size):
    selected = EMPTY LIST
    FOR iteration FROM 1 TO k:
        tournament = RANDOM_SAMPLE(population, tournament_size)
        winner = INDIVIDUAL in tournament WITH MAXIMUM fitness
        ADD winner.clone() TO selected
    END FOR
    RETURN selected
END FUNCTION

FUNCTION tournament_selection_probabilistic(population, k, tournament_size):
    selected = EMPTY LIST
    FOR iteration FROM 1 TO k:
        tournament = RANDOM_SAMPLE(population, tournament_size)
        total_tournament_fitness = SUM( f for each individual in tournament )
        r = RANDOM_NUMBER between 0 and total_tournament_fitness
        cumulative = 0
        FOR each individual in tournament:
            cumulative += individual.fitness
            IF cumulative >= r THEN
                ADD individual.clone() TO selected
                BREAK
            END IF
        END FOR
    END FOR
    RETURN selected
END FUNCTION

  - `def ranking_selection(population, k)`: Implement ranking-based selection.

  FUNCTION ranking_selection(population, k):
    sorted_population = SORT(population, BY fitness ascending)
    N = LENGTH(sorted_population)
    total_rank = SUM( rank for rank FROM 1 TO N )  # = N(N+1)/2
    selected = EMPTY LIST
    FOR iteration FROM 1 TO k:
        r = RANDOM_NUMBER between 0 and total_rank
        cumulative = 0
        FOR rank FROM 1 TO N:
            cumulative += rank
            IF cumulative >= r THEN
                ADD sorted_population[rank-1].clone() TO selected
                BREAK
            END IF
        END FOR
    END FOR
    RETURN selected
END FUNCTION

  - `def boltzmann_selection(population, k, temperature=1.0)`: Implement Boltzmann selection.
  FUNCTION boltzmann_selection(population, k, temperature):
    weights = [ EXP(individual.fitness / temperature) for each individual in population ]
    total_weight = SUM(weights)
    selected = EMPTY LIST
    FOR iteration FROM 1 TO k:
        r = RANDOM_NUMBER between 0 and total_weight
        cumulative = 0
        FOR each individual, weight in population, weights:
            cumulative += weight
            IF cumulative >= r THEN
                ADD individual.clone() TO selected
                BREAK
            END IF
        END FOR
    END FOR
    RETURN selected
END FUNCTION

  - `def universal_selection(population, k)`: Implement universal selection.
  FUNCTION universal_selection(population, k):
    total_fitness = SUM( f for each individual in population )
    step = total_fitness / k
    start = RANDOM_NUMBER between 0 and step
    pointers = [ start + i * step for i FROM 0 TO k-1 ]
    
    selected = EMPTY LIST
    FOR each pointer in pointers:
        cumulative = 0
        FOR each individual in population:
            cumulative += individual.fitness
            IF cumulative >= pointer THEN
                ADD individual.clone() TO selected
                BREAK
            END IF
        END FOR
    END FOR
    RETURN selected
END FUNCTION

  - `def get_selection_method(method_name, **kwargs)`: Return the appropriate selection function (wrapped as a lambda) based on `method_name` and extra parameters.
- **Configurability:**  
  Use `get_selection_method()` to choose the desired method in `ga_engine.py` by passing a string (e.g., `"tournament"`) and additional parameters via a dictionary.

---

### 3.5. `fitness.py`
- **Purpose:** Provide functions to evaluate the fitness of a `TriangleIndividual`.
- **Responsibilities:**
  - Render the individual’s phenotype.
  - Compare the rendered image to the target image using an error metric (e.g., Mean Squared Error, MSE).
  - Convert the error to a fitness value (e.g., `fitness = 1 / (1 + error)`).
- **Required Methods:**
  - `def compute_mse(image1, image2)`: Helper function that calculates the mean squared error between two images (numpy arrays).
  - `def compute_triangle_fitness(individual, target_image)`: Use `individual.render()` to obtain the phenotype, convert both images to grayscale, compute MSE, and return the fitness value.
  FUNCTION compute_mse(image_target, image_rendered):
    # image_target and image_rendered are matrices (arrays) of equal dimensions.
    error_sum = 0
    FOR i FROM 1 TO M:
        FOR j FROM 1 TO N:
            error_sum += ( image_target[i][j] - image_rendered[i][j] )^2
        END FOR
    END FOR
    mse = error_sum / (M * N)
    RETURN mse
END FUNCTION

FUNCTION compute_triangle_fitness(individual, target_image):
    rendered_image = individual.render()   # Render phenotype using individual's genotype
    # Convert both images to grayscale arrays of size MxN
    target_gray = CONVERT_TO_GRAYSCALE(target_image)
    rendered_gray = CONVERT_TO_GRAYSCALE(rendered_image)
    
    mse_value = compute_mse(target_gray, rendered_gray)
    fitness = 1 / (1 + mse_value)
    RETURN fitness
END FUNCTION

---

### 3.6. `termination.py`
- **Purpose:** Contains termination conditions for the GA loop.
- **Responsibilities:**
  - Terminate after a maximum number of generations.
  - Terminate if the best fitness improvement over a specified window is below a threshold.
  - Terminate if population diversity (computed from fitness standard deviation) is too low.
- **Required Functions:**
  - `def stop_after_max_generations(current_generation, max_generations)`: Return True if the current generation is ≥ max_generations.
  - `def stop_if_stagnant(best_fitness_history, window_size, stagnation_threshold)`: Return True if improvement is less than the threshold over a sliding window.
  FUNCTION stop_if_stagnant(best_fitness_history, window_size, stagnation_threshold):
    IF LENGTH(best_fitness_history) < window_size THEN
        RETURN False  # Not enough data to decide
    END IF

    recent_fitness = LAST window_size elements of best_fitness_history
    improvement = MAX(recent_fitness) - MIN(recent_fitness)
    
    IF improvement < stagnation_threshold THEN
        RETURN True
    ELSE
        RETURN False
    END IF
END FUNCTION
  - `def compute_diversity(population)`: Calculate a diversity metric based on fitness values.
  - `def structure_convergence(population, diversity_threshold)`: Return True if diversity is below the threshold.
  FUNCTION compute_diversity(population):
    fitnesses = [ f for each individual with fitness f in population ]
    mean_fitness = SUM(fitnesses) / LENGTH(fitnesses)
    variance = SUM( (f - mean_fitness)^2 for each f in fitnesses ) / LENGTH(fitnesses)
    diversity = SQRT(variance)
    RETURN diversity
END FUNCTION

FUNCTION structure_convergence(population, diversity_threshold):
    diversity = compute_diversity(population)
    IF diversity < diversity_threshold THEN
        RETURN True
    ELSE
        RETURN False
    END IF
END FUNCTION
  - `def check_termination(current_generation, max_generations, best_fitness_history, window_size, stagnation_threshold, population, diversity_threshold)`: Combine conditions and return True if any termination condition is met.

---

### 3.7. `utils.py`
- **Purpose:** Contains helper functions used across the project.
- **Responsibilities:**
  - Load images (using Pillow).
  - Resize images if needed.
  - Save output images.
  - Plot fitness curves.
  - Log run details.
- **Methods:**  
  Define helper functions such as:
  - `def load_image(path)`
  - `def resize_image(image, width, height)`
  - `def save_image(image, path)`
  - `def plot_fitness_curve(fitness_history, output_path)`

---

## 4. Configuration (config.yml)

The `config.yml` file should define:
- **General Parameters:**  
  - `problem`: `"triangles"`
  - `input_image`: Path to the target image (e.g., `"data/input_image.jpg"`)
- **GA Parameters:**  
  - `population_size`: e.g., `100`
  - `num_generations`: e.g., `500`
  - `mutation_rate`: e.g., `0.01`
  - `crossover_rate`: e.g., `0.9`
  - `num_triangles`: e.g., `50`
  - `canvas_size`: e.g., `[256, 256]`
- **Selection Parameters:**  
  - `selection_method`: e.g., `"tournament"`
  - `selection_params`:  
    - `tournament_size`: e.g., `5`
    - `deterministic`: `true`
- **Termination Parameters:**  
  - `max_generations`: Same as `num_generations`
  - `stagnation_window_size`: e.g., `10`
  - `stagnation_threshold`: e.g., `0.001`
  - `diversity_threshold`: e.g., `0.1`

---
