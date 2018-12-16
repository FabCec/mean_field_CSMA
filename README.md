# MEAN-FIELD LIMITS FOR LARGE-SCALE RANDOM-ACCESS NETWORKS

The code in this repository support the results in the paper "Mean-Field Limits for Large-Scale Random-Access Networks".

Features currently availables:

- Simulate CSMA network -> Stochastic model denoted X^{(N)}(t)

- Solve differential equation (3.1) limit of X^{(N)}(t) as N goes to infinity 

INSTRUCTION:

In main.py change the input section

- G : (np.array) Interference graph
- t_rates, b_rates, a_rates : (list) transmission, back-off, and arrival rates
- T : Stopping time of the simulation

- N : Number of nodes in each class
- num_steps : Number of steps in the numerical solution of the differential equation

TO DO:

- Add specific proportion of nodes in each class. Currently N in each class
- Keep track of waiting time


