import numpy as np
import pandas as pd

from functions import find_c, get_pi_backoff


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
