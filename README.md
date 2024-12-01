# Masters-Data-Science-DS785-Capstone-VRP-mTSP

This repository contains the code and resources for the DS785 Capstone Project, focusing on solving a manufacturing-styled Vehicle Routing Problem (VRP) or Multi-Traveling Salesman Problem (mTSP) using synthetic data, discrete event simulations, and mathematical optimization. Below is a guide to the key files and their purposes.

## Key Files
1. `SyntheticData+ProblemPrep_FINAL.ipynb` is the way problems are synthetically generated within the active directory by creating a `ProblemSets` folder and directory for the use of the rest of the code.
2. `DiscreteEventSimulation.py` is how discrete event simulations (DES) are carried out in parallel, given arguments passed to the script.
3. `MainFile.ipynb` is the primary running code that orchestrates the problem data, the simulations running and reading in parallel, and visualizations and statistical comparisons.
4. `MathOpt.ipynb` is the mathematical optimization file used to run Mixed Integer Linear Programming (MILP) using a Windows Subsystem for Linux (WSL2) setup (not required) and an academic license with Gurobi (currently as is coded, would need to be swapped out for alternative solvers such as HiGHS, CBC, etc.)

## Directory snapshots
The key files operate within a folder directory system to allow `multiprocessing` more fully. The screenshots below show my local directory after the files are completely run.

### Data Results in dataframes
![image](https://github.com/user-attachments/assets/4f1ee766-eb9a-4396-bef2-44f8131906c2)

### Screenshot of directory created (\ProblemSets)
![image](https://github.com/user-attachments/assets/1c5ab0d5-ccfd-4298-9863-7dac33481ebc)

### \ProblemSets\Training_Queue75
![image](https://github.com/user-attachments/assets/4bee1a0a-3231-41ef-823c-f04c9140bd07)

### \ProblemSets\Training_Queue75\Solutions
![image](https://github.com/user-attachments/assets/90ca7257-ec25-452d-9e10-341925a42ab0)

### \ProblemSets\Training_Queue75\Solutions\Traditional_true_SPT
![image](https://github.com/user-attachments/assets/dbd48aea-adb2-49d3-b5ae-3e9b293fb6da)
