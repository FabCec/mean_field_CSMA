import matplotlib.pyplot as plt
import networkx as nx


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


def plot_occupancy_compare(sol_truth, sol_sim, c, max_store, xi=None, T=10, path_output=None):

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

    if path_output:
        plt.savefig(path_output + "/Stochastic_Class_{}.png".format(c))
    else:
        plt.show()


def plot_interference_graph(G, path_output=None):
    # Plot interference graph
    C = G.shape[0]
    nx_G = nx.from_numpy_matrix(G)
    nx.draw(nx_G, arrows=False, node_size=500, labels={n: n + 1 for n in range(C)}, pos=nx.kamada_kawai_layout(nx_G))
    if path_output:
        plt.savefig(path_output + "/Interference_Graph.png")
    else:
        plt.show()
