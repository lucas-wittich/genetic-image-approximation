# TP2 - Genetic Algorithms
**Artificial Intelligence Systems – ITBA (2025)**  
Authors: [Your Name(s)]  
Date: [Submission Date]  

---

## 📘 Project Description

This project implements a Genetic Algorithm engine to solve two visual approximation problems:

### Exercise 1: ASCII Image Generation



Given a square input image, the goal is to generate an ASCII representation arranged in an NxN grid that visually resembles the original image. This is achieved by applying a Genetic Algorithm (GA) to optimize the placement of ASCII characters based on brightness values.

#### Image Preprocessing:
Since the input is already a square image, we begin by converting it to grayscale. Each pixel is assigned a single brightness value ranging from 0 (black) to 255 (white), which simplifies the process of mapping pixel intensities to ASCII characters.

#### ASCII Character Set:
We define a set of ASCII characters ordered from darkest to lightest based on perceived brightness. Characters like `@`, `#`, and `%` represent dark regions, while `-`, `.`, and space (` `) are used for light regions. Each character is assigned a brightness value to support accurate mapping to grayscale levels.

#### Individual Representation:
In our GA, each individual represents one possible ASCII version of the image. It is modeled as an NxN grid where each cell (gene) contains a single ASCII character. The initial population is randomly generated, ensuring diverse starting points for the algorithm to explore.

#### Fitness Function:
The fitness function evaluates how closely a candidate ASCII image matches the original grayscale image:
- Each ASCII character’s brightness is compared to the corresponding pixel intensity.
- The absolute or squared difference is computed at every position.
- A lower total error indicates a better match. The fitness score is defined as the inverse of the error or as a negative error value.

#### Selection Mechanisms:
To choose individuals for reproduction, we use methods such as:
- Tournament selection: Select the best among a random subset.
- Roulette wheel selection: Probability of selection is proportional to fitness.

These strategies prioritize individuals with higher fitness while maintaining diversity.

#### Crossover Operators:
New individuals are produced by combining parts of two parents:
- One-point crossover (e.g. half of the rows from each parent).
- Alternating rows or columns.

Crossover helps propagate beneficial traits by recombining solutions.

#### Mutation:
Mutation introduces random variation to maintain genetic diversity:
- One or more ASCII characters are randomly replaced.
- Mutation is applied with low probability to avoid disrupting promising solutions.

#### Generational Iteration:
The algorithm evolves over several generations. In each generation, it performs fitness evaluation, selection, crossover, and mutation to create new candidate solutions.

#### Termination Criteria:
The algorithm stops when one of the following is true:
- A maximum number of generations is reached.
- There is little to no improvement over several generations.
- A predefined error threshold is achieved.

#### Final Output:
The best-performing individual from the final generation is returned as the ASCII image that most closely resembles the original input image.


### Exercise 2: Image Approximation with Triangles
Given any input image and a triangle count **T**, approximate the image by drawing T translucent, colored triangles over a blank canvas.

---

## 🧠 Problem Inputs & Outputs

### Inputs
- Input image file
- Number of triangles `T` (for Exercise 2)
- Genetic algorithm hyperparameters:
  - Population size
  - Crossover and mutation probabilities
  - Selection / crossover / mutation methods
  - Termination criteria (generations, fitness stagnation, etc.)

### Outputs
- Generated image (ASCII art or triangle-based)
- A list of triangles (position, color, opacity)
- Performance metrics:
  - Best fitness
  - Error over generations
  - Number of generations
  - Execution time (optional)

---

## ⚙️ How to Run


1. Clone this repository:
   ```sh
   git clone https://github.com/lucas-wittich/72.27-SIA
   cd 72.27-SIA/TP1/Sokoban
   ```

2. Run the code:

   ```sh
   python <file-to-run.py>
   ```

   - [`main.py`](src/main.py)  
   Run the main method for the GA_engine.
   Optional: Pass your own config.json for an image. See [`config.json`](configs/config.json) for default config.
   ```sh
   python main.py <config_file.json> 
   ```

3. Examine the output and metrics in [`data/outputs/<image_name>`](data/outputs)