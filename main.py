import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def find_c(distr):
    distr_norm = distr/np.sum(distr)
    ind = np.random.choice(range(len(distr)), p=distr_norm)
    return ind


def get_independent_sets(G):

    curr_ind = 0
    indep_sets = [tuple([0]*G.shape[0])]
    set_indep_sets = set(indep_sets)

    while curr_ind < len(indep_sets):
        curr_set = list(indep_sets[curr_ind])
        curr_ind = curr_ind + 1
        unblocked = np.matmul(G, curr_set) == 0
        for ind in range(G.shape[0]):
            if unblocked[ind]:
                aux_set = curr_set.copy()
                aux_set[ind] = 1
                if tuple(aux_set) not in set_indep_sets:
                    indep_sets.append(tuple(aux_set))
                    set_indep_sets = set(indep_sets)

    return set(indep_sets)


def get_pi_activity(G, act_factor, return_sets=False, normalize=True):
    ind_sets = list(get_independent_sets(G))
    ind_sets = sorted(ind_sets, key=np.sum)
    pi_act = []

    for s in ind_sets:
        aux_prod = 1
        for c in range(G.shape[0]):
            if s[c]:
                aux_prod = aux_prod * act_factor[c]
        pi_act.append(aux_prod)

    if not normalize:
        return pi_act

    if return_sets:
        return pi_act / np.sum(pi_act), ind_sets

    return pi_act / np.sum(pi_act)


def get_pi_transm(G, act_factor, normalize=True):

    ind_sets = list(get_independent_sets(G))
    ind_sets = sorted(ind_sets, key=np.sum)

    pi_act = get_pi_activity(G, act_factor, normalize=normalize)

    pi_transm = np.zeros(G.shape[0])

    for ind, s in enumerate(ind_sets):
        for c in range(G.shape[0]):
            if s[c]:
                pi_transm[c] += pi_act[ind]

    return pi_transm


def get_pi_backoff(G, act_factor):

    ind_sets = list(get_independent_sets(G))
    ind_sets = sorted(ind_sets, key=np.sum)

    pi_act = get_pi_activity(G, act_factor)

    pi_back = np.zeros(G.shape[0])

    for ind, s in enumerate(ind_sets):
        unblocked = np.matmul(G + np.eye(G.shape[0]), s) == 0
        for c in range(G.shape[0]):
            if unblocked[c]:
                pi_back[c] += pi_act[ind]

    return pi_back


def get_xi_complete_interference(a_rates, b_rates, t_rates):

    load = np.sum([a_rates[i]/t_rates[i] for i in range(len(a_rates))])

    return [a_rates[i]/(b_rates[i]*(1-load)) for i in range(len(a_rates))]


def get_xi_iterative(G, a_rates, b_rates, t_rates):

    C = G.shape[0]
    sig = b_rates/t_rates
    xi = [0.1]*len(a_rates)
    rho = a_rates/t_rates

    for iter in range(100):
        act_factor = sig*xi
        Z = np.sum(get_pi_activity(G, act_factor, return_sets=False, normalize=False))
        T = get_pi_transm(G, act_factor, normalize=False)/xi
        for c in range(C):
            xi[c] = rho[c]*Z/T[c]

    return xi


def run_diff_eq(t_rates, b_rates, a_rates,
                G, T=1, max_store=3, num_steps=1000):

    C = G.shape[0]
    step_size = T/num_steps
    columns = ['Time'] + ['x_{}_{}'.format(c, n) for c in range(C) for n in range(max_store)]
    occupancy = pd.DataFrame(columns=columns)

    sol = np.zeros([C, num_steps + 2])
    for c in range(C):
        sol[c, 0] = 1

    perc = 10
    check_point = num_steps * (perc / 100)

    for step in range(num_steps):
        if step > check_point:
            print("{}% done - MSE current variation {} ".format(perc, np.sqrt(np.sum(np.array(curr_variation)**2))))
            perc = perc + 10
            check_point = num_steps * (perc / 100)

        act_factor = [b_rates[c]*(1 - sol[c, 0]) / t_rates[c] for c in range(C)]
        pi_back = get_pi_backoff(G, act_factor=act_factor)

        tot_variation = []
        for c in range(C):
            plus_rate = a_rates[c]
            minus_rate = b_rates[c]*pi_back[c]
            curr_variation = []
            arr_variation = []
            back_variation = []
            for i in range(step + 2):
                arr_variation.append(plus_rate*sol[c, i])
                back_variation.append(minus_rate*sol[c, i])

            curr_variation.append(back_variation[1] - arr_variation[0])
            tot_variation.append(back_variation[1] - arr_variation[0])
            for i in np.arange(1, step + 1):
                curr_variation.append(arr_variation[i-1] - back_variation[i] - arr_variation[i] + back_variation[i+1])

            curr_variation.append(-np.sum(curr_variation))

            assert len(curr_variation) == step+2, "cv {}, step+1 {}".format(len(curr_variation), step+2)

            for i in range(len(curr_variation)):
                sol[c, i] += curr_variation[i]*step_size

            sol[c, :] = sol[c, :]/np.sum(sol[c, :])

        new_row = [step*step_size] + [sol[c][n] for c in range(C) for n in range(max_store)]
        new_df = pd.DataFrame([new_row], columns=columns)
        occupancy = occupancy.append(new_df)

    occupancy = occupancy.set_index('Time')

    return occupancy


def run(t_rates, b_rates, a_rates,
        G, T=10, N=10, max_store=3):
    """
    Run CSMA simulator till time T.
    Statistics of each node are in t_vec, b_vec, a_vec
    G is an adjacency matrix

    :param t_vec:
    :param b_vec:
    :param a_vec:
    :param G:
    :param T:
    :return:
    """

    C = len(t_rates)

    np.random.seed(seed=1)

    time = 0
    active = [False]*C

    max_queue = 4
    queue_pop = np.zeros([C, max_queue])
    # queue_pop[c, n] = # of class c nodes with n packets in buffer
    for c in range(C):
        queue_pop[c, 0] = N

    columns = ['Time'] + ['X_{}_{}'.format(c, n) for c in range(C) for n in range(max_store)]
    occupancy = pd.DataFrame(columns=columns)

    arrival_ave = 1/a_rates
    transm_ave = 1/t_rates

    perc = 10
    check_point = N*T*(perc/100)
    while time < N*T:
        if time > check_point:
            print("{}% done".format(perc))
            perc = perc + 10
            check_point = N * T * (perc / 100)

        next_actions = np.inf*np.ones([3, C])
        back_ave = N/(b_rates * (N - queue_pop[:, 0]))
        blocked = np.matmul(G, active) > 0
        backoff = [not blocked[c] and not active[c] for c in range(C)]

        # Find next action
        for c in range(C):
            next_actions[0, c] = np.random.exponential(arrival_ave[c])
            if active[c]:
                next_actions[1, c] = np.random.exponential(transm_ave[c])
            if backoff[c]:
                next_actions[2, c] = np.random.exponential(back_ave[c])

        # Update time
        time += np.min(next_actions)

        # Update occupancy
        new_row = [time] + [queue_pop[c][n]/N for c in range(C) for n in range(max_store)]
        new_df = pd.DataFrame([new_row], columns=columns)
        occupancy = occupancy.append(new_df)

        # Update
        event = np.unravel_index(np.argmin(next_actions, axis=None), next_actions.shape)
        curr_e = event[0]
        curr_c = event[1]
        curr_distr = queue_pop[curr_c, :]

        # New arrival
        if curr_e == 0:
            select_n = find_c(curr_distr)
            if queue_pop[curr_c, select_n] <= 0:
                print('Error A', select_n, queue_pop[curr_c, :])

            if select_n == queue_pop.shape[1]-1:
                queue_pop_aux = queue_pop
                shape_old_queue_pop = queue_pop.shape
                print('max_pop doubled to {}'.format(shape_old_queue_pop[1]*2))
                queue_pop = np.zeros([shape_old_queue_pop[0], shape_old_queue_pop[1]*2])
                queue_pop[:, :shape_old_queue_pop[1]] = queue_pop_aux
            queue_pop[curr_c, select_n] -= 1
            queue_pop[curr_c, select_n + 1] += 1

        # New transmission
        if curr_e == 1:
            active[curr_c] = False

        # New back-off
        if curr_e == 2:
            active[curr_c] = True
            select_n = find_c(curr_distr[1:]) + 1
            if queue_pop[curr_c, select_n] <= 0:
                print('Error BO', select_n, queue_pop[curr_c, :])
            queue_pop[curr_c, select_n] -= 1
            queue_pop[curr_c, select_n - 1] += 1

    occupancy['Time'] = occupancy['Time']/N
    occupancy = occupancy.set_index('Time')

    return occupancy


def plot_occupancy(occupancy, c, max_store):

    C = int(occupancy.shape[1]/max_store)
    true_vec = [True]*max_store
    false_vec = [False]*max_store
    selected = []
    for c_curr in range(C):
        if c_curr == c:
            selected = selected + true_vec
        else:
            selected = selected + false_vec

    df = occupancy.iloc[:, selected]

    df.plot()
    plt.show()


def plot_occupancy_compare(sol_truth, sol_sim, c, max_store, xi=None, T=10):

    C = int(sol_truth.shape[1] / max_store)
    true_vec = [True]*max_store
    false_vec = [False]*max_store
    selected = []
    for c_curr in range(C):
        if c_curr == c:
            selected = selected + true_vec
        else:
            selected = selected + false_vec

    df_1 = sol_truth.iloc[:, selected]
    df_2 = sol_sim.iloc[:, selected]

    ax = df_1.plot(linestyle='--', cmap=plt.cm.tab10)  # viridis
    df_2.plot(ax=ax, cmap=plt.cm.tab10)

    if xi is not None:
        equilibrium = []
        for m in range(max_store):
            equilibrium.append((1-xi)*xi**m)
            ax.plot([0, T], [equilibrium[m], equilibrium[m]])

    plt.show()



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
T = 15

# Number of coordinates of the occupancy measure to store
# Final dataframe will have C*max_store columns
max_store = 4

#######################
###### END INPUT ######
#######################


C = G.shape[0]

# Check dimensions
assert G.shape[1] == C, "Wrong Graph G"
assert len(t_rates) == C, "Check dimension transmission rates"
assert len(a_rates) == C, "Check dimension arrival rates"
assert len(b_rates) == C, "Check dimension back-off rates"

# Plot interference graph
nx_G = nx.from_numpy_matrix(G)
nx.draw(nx_G, arrows=False, node_size=500, labels={n: n+1 for n in range(C)}, pos=nx.kamada_kawai_layout(nx_G))
plt.show()

# Print activity factors
xi = get_xi_iterative(G, a_rates, b_rates, t_rates)
print("Activity factors", get_xi_iterative(G, a_rates, b_rates, t_rates))

# Solve differential equation
print('\n**** Solve Differential Equation ****\n')
occupancy_eq = run_diff_eq(t_rates=t_rates, a_rates=a_rates, b_rates=b_rates,
                           T=T, G=G, max_store=max_store, num_steps=500)

# Run the simulation
print('\n**** Run simulation ****\n')
occupancy = run(t_rates=t_rates, a_rates=a_rates, b_rates=b_rates, N=500, T=T, G=G, max_store=max_store)

# Plot the results
print('\n**** Plot ****\n')
for c in range(len(t_rates)):
    plot_occupancy_compare(sol_truth=occupancy_eq, sol_sim=occupancy, c=c, max_store=max_store, T=T)

