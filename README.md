# Metaheuristic Optimization and Interpretability Project


This repository contains code and experiments exploring metaheuristic algorithms for combinatorial optimization and the interpretability of their solutions using decision tree surrogates.

## Algorithms Implemented

- **Ant Colony Optimization (ACO)**
- **Biogeography-Based Optimization (BBO)**
- **Bee Reproduction Optimization (BRO)**
- **Chemical Reaction Optimization (CRO)**
- **Particle Swarm Optimization (PSO)**
- **Random Search, Simulated Annealing, Tabu Search, Genetic Algorithm** (additional baselines)

All algorithms are integrated using the [MEALPY](https://github.com/thieu1995/mealpy) Python library.

## Problem Domains

- Satisfiability Problems (SAT)
- Graph Coloring (GC)
- Staff Rostering (BT)
- Other combinatorial and constraint satisfaction benchmarks

## Interpretability with Decision Trees

To make metaheuristic solutions more transparent, this project fits decision tree models as surrogates to explain and visualize the fitness landscape and search behavior of each optimization method.

- Surrogate trees are trained on evaluated solutions.
- Scatterplots show correlation between true fitness and tree-predicted fitness, providing insight into the quality of approximation and generalization.

## Project Structure

- `notebooks/` — Jupyter notebooks for algorithm execution, training surrogates, and plotting analysis.
- `src/` — Python source files for algorithms, utility scripts, and data processing.
- `data/` — Benchmark instances and results (JSON, CSV).
- `plots/` — Generated figures and comparison plots.

This repository owned by Uma Shree.
