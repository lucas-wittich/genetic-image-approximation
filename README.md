# Genetic Image Approximation.
---

## Features
- Modular GA components: selection, crossover, mutation
- Fitness: error vs. target image **MSE/PSNR** for triangles
- Reproducible runs (fixed seeds), per-generation logs, and evolution **GIFs** (triangles)

---
## Problem Inputs & Outputs

### Inputs
Config containing the at least the following, defaults to ['config.json'](configs/config.json) if none provided.
- Input image file
- Number of triangles `T`
- Genetic algorithm hyperparameters:
  - Population size
  - Crossover and mutation probabilities
  - Selection / crossover / mutation methods
  - Termination criteria (generations, fitness stagnation, etc.)

### Outputs
- Generated image
- A list of triangles (position, color, opacity)
- A GIF of the evolution
- Performance metrics:
  - Best fitness
  - Fitness over generations (Plot and CSV file)
  - Number of generations

---

## ⚙️ How to Run


1. Clone this repository:
   ```sh
   git clone https://github.com/lucas-wittich/genetic-image-approximation
   cd Sokoban
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