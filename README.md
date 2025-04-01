# TP2 - Genetic Algorithms
**Artificial Intelligence Systems – ITBA (2025)**  
Authors: [Your Name(s)]  
Date: [Submission Date]  

---

## 📘 Project Description

This project implements a Genetic Algorithm engine to solve two visual approximation problems:

### Exercise 1: ASCII Image Generation
Given a square input image, approximate it using ASCII characters arranged in an NxN grid.

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

1. Install dependencies:
```bash
pip install -r requirements.txt
