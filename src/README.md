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
