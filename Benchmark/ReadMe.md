# Graph-based Metamodeling (GraphMM) for Multi-scale Biological Systems - Benchmark toy system

This project implements a graph-based metamodeling approach for multi-scale biological systems, focusing on the integration of surrogate models for islet and cell-level insulin dynamics using a simplified toy system.

## Project Structure

- `InputModel/`: Contains groundtruth and subsystem models
- `GraphMetamodel/`: Includes the surrogate model and coupling graph definitions
- `results/`: Stores output files from model simulations

## Key Components

1. **Surrogate Models**: 
   - Model A: Islet-level insulin dynamics
   - Model B: Cell-level insulin dynamics

2. **Coupling Graph**: 
   - Defines connections between surrogate models
   - Implements multi-scale inference

3. **Metamodel**: 
   - Integrates surrogate models using the coupling graph
   - Performs inference across different scales

## Main Script: enumerate_metamodel.py

This script is the core of the project, performing the following tasks:

1. Imports necessary modules and data
2. Loads surrogate model states
3. Iterates through different parameter combinations (ω1 and ω2)
4. Constructs coupling graphs for each combination
5. Performs multi-scale inference
6. Saves results to CSV files

## Data Visualization

The script also includes functionality to visualize the results:

- Reads all generated CSV files
- Plots time series data for different model components

## Usage

To run the metamodel enumeration:

1. Ensure all required dependencies are installed
2. Run `enumerate_metamodel.py`
3. Results will be saved in the `results/enumerate_metamodel_v2/` directory
4. Visualizations can be generated using the plotting functions in the script

<<<<<<< Updated upstream
## Note

This README provides an overview of the project structure and main script functionality. For detailed information on specific functions and classes, please refer to the individual module docstrings and comments within the code.
=======
## Copyright


© 2023 GraphMM Project Contributors (chenxi.wang@salilab.org). All rights reserved.

This project and its contents are protected under applicable copyright laws. Unauthorized reproduction, distribution, or use of this material without express written permission from the GraphMM Project Contributors is strictly prohibited.

For inquiries regarding usage, licensing, or collaboration, please contact the project maintainers.

>>>>>>> Stashed changes
