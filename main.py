import numpy as np
import os

from functions import get_xi_iterative
from visualization import plot_interference_graph, plot_occupancy_compare
from simulation import run_diff_eq, run

###################
###### INPUT ######
###################


t_rates = np.array([10, 2, 2, 2, 2, 2, 2, 2, 2, 2])
a_rates = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
b_rates = np.array([10, 3, 3, 3, 3, 3, 3, 3, 3, 3])

"""
# Star Shape
C = 7
G = np.zeros([C, C])
for c in np.arange(1, C):
    G[0, c] = 1
    G[c, 0] = 1  #"""

"""
# Complete Graph
C = 7
G = np.ones([C, C])
for c in range(C):
    G[c, c] = 0  #"""

"""
# Linear Graph
C = 7
G = np.zeros([C, C])
for c in np.arange(1, C-1):
    G[c, c+1] = 1
    G[c, c-1] = 1
G[0,1] = 1
G[C-2, C-1] = 1#"""

"""
# Circular Graph
C = 7
G = np.zeros([C, C])
for c in range(C):
    G[c, np.mod(c+1, C)] = 1
    G[c, np.mod(c-1, C)] = 1  # """

#"""
# Customized graph
G = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
              ])  #"""

# Time till the simulation run
T = 20

# Num_steps in the numerical solution of the differential equation
num_steps = 1000

# Number of nodes per class
N = 1000

# Number of coordinates of the occupancy measure to store
# Final dataframe will have C*max_store columns
max_store = 4

# Saving directory
path_output = os.getcwd() + "/10Nodes"

#######################
###### END INPUT ######
#######################


C = G.shape[0]

# Check dimensions
assert G.shape[1] == C, "Wrong Graph G"
assert len(t_rates) == C, "Check dimension transmission rates"
assert len(a_rates) == C, "Check dimension arrival rates"
assert len(b_rates) == C, "Check dimension back-off rates"


# Print activity factors
xi = get_xi_iterative(G, a_rates, b_rates, t_rates)
print("Activity factors", get_xi_iterative(G, a_rates, b_rates, t_rates))

# Solve differential equation
print('\n**** Solve Differential Equation ****\n')
occupancy_eq = run_diff_eq(t_rates=t_rates, a_rates=a_rates, b_rates=b_rates,
                           T=T, G=G, max_store=max_store, num_steps=num_steps)

# Run the simulation
print('\n**** Run simulation ****\n')
occupancy = run(t_rates=t_rates, a_rates=a_rates, b_rates=b_rates, N=N, T=T, G=G, max_store=max_store)

# Create directory for saving the results
os.makedirs(path_output, exist_ok=True)

# Plot and save the results
plot_interference_graph(G, path_output=path_output)
print('\n**** Plot ****\n')
for c in range(len(t_rates)):
    plot_occupancy_compare(sol_truth=occupancy_eq, sol_sim=occupancy,
                           c=c, max_store=max_store, T=T, path_output=path_output)

