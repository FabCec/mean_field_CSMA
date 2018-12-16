import numpy as np


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

