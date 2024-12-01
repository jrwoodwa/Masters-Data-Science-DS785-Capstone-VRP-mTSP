# Masters-Data-Science-DS785-Capstone-VRP-mTSP

This repository contains the code and resources for the DS785 Capstone Project, focusing on solving a manufacturing-styled Vehicle Routing Problem (VRP) or Multi-Traveling Salesman Problem (mTSP) using synthetic data, discrete event simulations, and mathematical optimization. Below is a guide to the key files and their purposes.

## Key Files
1. `SyntheticData+ProblemPrep_FINAL.ipynb` is the way problems are synthetically generated within the active directory by creating a `ProblemSets` folder and directory for the use of the rest of the code.
2. `DiscreteEventSimulation.py` is how discrete event simulations (DES) are carried out in parallel, given arguments passed to the script.
3. `MainFile.ipynb` is the primary running code that orchestrates the problem data, the simulations running and reading in parallel, and visualizations and statistical comparisons.
4. `MathOpt.ipynb` is the mathematical optimization file used to run Mixed Integer Linear Programming (MILP) using a Windows Subsystem for Linux (WSL2) setup (not required) and an academic license with Gurobi (currently as is coded, would need to be swapped out for alternative solvers such as HiGHS, CBC, etc.)
