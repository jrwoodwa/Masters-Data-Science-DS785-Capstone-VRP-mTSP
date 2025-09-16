# Masters-Data-Science-DS785-Capstone-VRP-mTSP

This repository contains the code and resources for a DS785 Capstone Project at the University of Wisconsin's master's data science program: “Smart Manufacturing Scheduling: Blending AI, Optimization, and Stochastic Modeling”.

The project used synthetic data, discrete event simulations, and mathematical optimization to address a manufacturing problem known as Parallel Job Shop Scheduling (PJSS), closely related to the Vehicle Routing Problem (VRP) or the Multi-Traveling Salesman Problem (mTSP). 

## Abstract

> At industrial bottlenecks, scheduling decisions often rely on guesswork or simplistic algorithms—overlooking resources and profits. 
> 
> This business case study blended AI, optimization, and uncertainty (i.e., stochastic modeling) to address two parallel-processing machines sharing a single queue, incorporating real-world factors such as sequence-dependent setups (i.e., asymmetry), machine-specific incapabilities, and unpredictable upstream queuing. 
> 
> Traditional dispatching scheduling treats each machine as independent, ignoring the advantages of cooperation. 
> 
> In contrast, this project’s cooperative approach optimized maximum flow time and average machine work time, thoroughly evaluated with synthetic yet lifelike problem sets. 
> 
> By integrating discrete event simulation, machine learning (ML), scalable optimization, and evaluating scheduling decisions iteratively (i.e., sequential policy), the approach achieved 20% improvements—reaching up to 60% in some scenarios—over traditional methods (and realized 10% gains even at larger 75-job queues), all within a real-time 10-second solve using a personal desktop. 
> 
> These results demonstrate to companies that meeting complex problems with decision science delivers significant practical improvements, showcasing its transformative power to monetize data and drive long-term competitive advantages.
> 
> Keywords: Smart manufacturing, operations scheduling, optimization, real-time, artificial intelligence (AI), machine learning (ML), stochastic modeling, sequential policy, robust optimization, discrete event simulation, prescriptive analytics, decision science, operations research, high-performance computing

## Key Files
0. `Smart Manufacturing Scheduling with Data Science_Woodward_5.57_FINAL.pdf` is the master's business case study paper. The five chapters covered Introduction (I), Literature Review (II), Methodology (III), Results (IV), and Discussion (V), with Chapter III diving into the technical parts.
1. `SyntheticData+ProblemPrep_FINAL.ipynb` is the way problems were synthetically generated within the active directory by creating a `ProblemSets` folder and directory for the use of the rest of the code.
2. `DiscreteEventSimulation.py` is how discrete event simulations (DES) were carried out in parallel, given arguments passed to the script.
3. `MainFile.ipynb` is the primary running code that orchestrated the problem data, ran the simulations in parallel, and did visualizations and statistical comparisons.
4. `MathOpt.ipynb` is the mathematical optimization file used to run Mixed Integer Linear Programming (MILP) using a Windows Subsystem for Linux (WSL2) setup (not required) and an academic license with Gurobi (currently coded as-is, would need to be refactored for alternative solvers such as HiGHS, CBC, etc.)

## Directory snapshots
The key files assume a folder directory system to enable `multiprocessing` or parallelizing computer threads more effectively. The screenshots below show my local directory after the files have completely run.

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
