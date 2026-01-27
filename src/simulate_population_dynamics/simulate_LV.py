import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import seaborn as sns
import pandas as pd
from copy import deepcopy
import sys
import os
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def compute_deltas(df, data_set):
    if data_set == "LV":
        df = df.sort_values(['species', 'census'])
        df['n_next'] = df.groupby('species')['n'].shift(-1)
        df['N_next'] = df.groupby('species')['N_t'].shift(-1)
        df['S_next'] = df.groupby('species')['S_t'].shift(-1)

        df['dn'] = df['n_next'] - df['n']
        df['dN'] = df['N_next'] - df['N_t']
        df['dS'] = df['S_next'] - df['S_t']
    else:
        df = df.sort_values(['species', 'treeID', 'census'])
        df['n_next'] = df.groupby('species')['n'].shift(-1)
        df['e_next'] = df.groupby('species')['e'].shift(-1)
        df['N_next'] = df.groupby('species')['N_t'].shift(-1)
        df['S_next'] = df.groupby('species')['S_t'].shift(-1)
        df['E_next'] = df.groupby('species')['E_t'].shift(-1)

        df['dn'] = df['n_next'] - df['n']
        df['de'] = df['e_next'] - df['e']
        df['dN'] = df['N_next'] - df['N_t']
        df['dS'] = df['S_next'] - df['S_t']
        df['dE'] = df['E_next'] - df['E_t']

    df.dropna(inplace=True)

    return df


def f(n, t, growth_rates, alpha):
    n = np.clip(n, 0, None)
    return n * growth_rates * (1 - alpha @ n)


def plot_solutions(sol, tspan, model="", iter=""):
    """
    Plots time-series solutions of a dynamical system, grouping species by variable type (X, Y, Z).

    Parameters:
    - sol: np.ndarray, shape (time, species), the solution matrix
    - tspan: array-like, the time vector
    - model: str, optional title suffix and filename tag
    """
    group_size = sol.shape[1] // 3
    group_colors = sns.color_palette("Set1", 3)  # Colors for X, Y, Z

    # Assign colors based on species group
    species_colors = (
        [group_colors[0]] * group_size +  # X
        [group_colors[1]] * group_size +  # Y
        [group_colors[2]] * group_size    # Z
    )

    # Fixed figure size
    plt.figure(figsize=(10, 6))

    # Aesthetic settings
    sns.set(style="whitegrid", context="notebook", font_scale=2.0)
    plt.figure(figsize=(10, 6))

    for i in range(sol.shape[1]):
        plt.plot(tspan, sol[:, i], lw=1.8, alpha=0.9, color=species_colors[i])

    # Fixed y-axis limits
    plt.ylim(0, 300)

    # Group legend
    legend_handles = [
        Patch(color=group_colors[2], label=r"$Z$"),
        Patch(color=group_colors[1], label=r"$Y$"),
        Patch(color=group_colors[0], label=r"$X$")
    ]
    ax = plt.gca()
    ax.set_position([0.12, 0.12, 0.78, 0.78])  # absolute fixed axis box

    plt.legend(
        handles=legend_handles,
        title="Genus",
        frameon=False,
        fontsize=16,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.88),  # (x, y) in figure coordinates
        borderaxespad=0,
    )

    # Titles and labels with LaTeX formatting
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Population Size (n)", fontsize=20)

    # Grid and layout
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    sns.despine()
    #plt.tight_layout()

    # Save figure
    path = f"dynamics_{model}_{iter}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close('all')
    #plt.show()


def create_df(solutions):
    T, S = solutions.shape

    # Increase the observation interval by only keeping every 10th time step
    # so that dn is not so small
    sampled_censuses = np.arange(T)[::50]
    subsamples = solutions[sampled_censuses]
    T = subsamples.shape[0]

    # Create base DataFrame
    df = pd.DataFrame({
        "census": np.repeat(np.arange(1, T + 1), S),
        "species": np.tile(np.arange(S), T),
        "n": subsamples.flatten()
    })

    # Compute N (total individuals) and S (species richness) over time
    total_N = subsamples.sum(axis=1)  # Total individuals at each time
    richness_S = (subsamples > 0).sum(axis=1)  # Species with n > 0

    # Broadcast totals into long format
    df["N_t"] = np.repeat(total_N, S)
    df["S_t"] = np.repeat(richness_S, S)

    dn_matrix = np.diff(subsamples, axis=0, append=np.zeros((1, S)))

    # Replace with -n_t if species goes extinct at t+1
    extinct_mask = subsamples[1:] == 0  # shape (T-1, S)
    dn_matrix[:-1][extinct_mask] = -subsamples[:-1][extinct_mask]

    # Flatten into dataframe
    df["dn"] = dn_matrix.flatten()

    # Compute forward differences for N and S: value(t+1) - value(t)
    dN = np.diff(total_N, append=0)
    dS = np.diff(richness_S, append=0)

    df["dN/S"] = np.repeat(dN/S, S)
    df["dS"] = np.repeat(dS, S)

    df = df[df['census'] < 20]

    return df

def create_df_clustered(solutions):
    T, S = solutions.shape

    # sample timepoints
    sampled_censuses = np.arange(T)[::50]
    subsamples = solutions[sampled_censuses]
    T = subsamples.shape[0]

    # assign species → cluster
    cluster_id = np.arange(S) // 12 + 1   # S species, clusters of size 12
    unique_clusters = np.unique(cluster_id)

    all_dfs = []

    for cluster in unique_clusters:

        # species indices in this cluster
        idx = np.where(cluster_id == cluster)[0]

        # extract cluster-specific subsample matrix
        sub = subsamples[:, idx]
        S_c = sub.shape[1]

        # compute N, S within cluster
        N_t = sub.sum(axis=1)
        S_t = (sub > 0).sum(axis=1)

        # dn: forward difference per species
        dn = np.diff(sub, axis=0, append=np.zeros((1, S_c)))

        # extinct-at-t+1 rule
        extinct_mask = sub[1:] == 0
        dn[:-1][extinct_mask] = -sub[:-1][extinct_mask]

        # dN and dS
        dN = np.diff(N_t, append=0)
        dS = np.diff(S_t, append=0)

        # build long dataframe for this cluster
        df_cluster = pd.DataFrame({
            "census": np.repeat(np.arange(1, T + 1), S_c),
            "species": np.tile(idx, T),        # original species IDs
            "cluster": cluster,
            "n": sub.flatten(),
            "N_t": np.repeat(N_t, S_c),
            "S_t": np.repeat(S_t, S_c),
            "dn": dn.flatten(),
            "dN/S": np.repeat(dN / S_c, S_c),
            "dS": np.repeat(dS, S_c),
        })

        # optional filtering
        df_cluster = df_cluster[df_cluster["census"] < 20]
        df_cluster = df_cluster[df_cluster['n'] > 0]

        all_dfs.append(df_cluster)

    # combine clusters back together
    return pd.concat(all_dfs, ignore_index=True)

def three_groups_LV(model_func="food_web", T=50, var=0.0, iter=""):
    S = 12
    N = 100

    noise_term = 0.0

    # Initialize matrix
    A = np.zeros((S, S))

    # Group boundaries
    group_size = S // 3
    group_indices = {
        'X': slice(0, group_size),
        'Y': slice(group_size, 2 * group_size),
        'Z': slice(2 * group_size, S)
    }

    def a_LV():
        """
        Constant interaction network
        """
        growth_rates = np.ones(S)

        # Equal competition for all species
        A = 0.125 * np.ones((S, S)) + np.random.normal(0, var, (S, S))

        # Intra-group competition: still per-individual (diagonal within group block)
        group_means = [0.25, 0.25, 0.25]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = (np.random.uniform
                (0,100,S))
        initial_conditions /= np.sum(initial_conditions)
        return growth_rates, A, initial_conditions

    def e_LV():
        """
        Food chain
        """
        growth_rates = np.array([1]*4 + [0.2]*4 + [-0.001]*4)

        # Inter-group competition with per-individual variability
        A[group_indices['Y'], group_indices['Z']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Z on Y
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Y on X

        A[group_indices['Z'], group_indices['Y']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Y on Z
        A[group_indices['Y'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # X on Y

        group_means = [0.25, 0.025, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1,55,4),
            np.random.uniform(1,30, 4),
            np.random.uniform(1, 15, 4)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def b_LV():
        """
        Two predators one prey
        """
        growth_rates = np.array([1.0]*4 + [-0.001]*8)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Z on X
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Y on X

        A[group_indices['X'], group_indices['Z']] = -0.025 + np.random.normal(0, var,(group_size, group_size)) # Z on X
        A[group_indices['X'], group_indices['Y']] = -0.025 + np.random.normal(0, var,(group_size, group_size)) # Y on X

        group_means = [0.25, 0.025, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1,65, 4),
            np.random.normal(1, 35, 8)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def c_LV():
        """
        One predator two prey
        """
        growth_rates = np.array([1.0] * 8 + [-0.001] * 4)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Y'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on Y

        A[group_indices['Z'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Z'], group_indices['Y']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on Y

        group_means = [0.25, 0.25, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1,65, 8),
            np.random.uniform(1, 35, 4)
        ])
        initial_conditions = np.clip(initial_conditions, 0.01, None)
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def f_LV():
        """
        Food chain with omnivory
        """
        growth_rates = np.array([1.0] * 4 + [0.2] * 4 + [-0.001] * 4)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Y'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Y on X

        A[group_indices['Z'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Z'], group_indices['Y']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['Y'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Y on X

        group_means = [0.25, 0.025, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1, 55, 4),
            np.random.uniform(1, 30, 4),
            np.random.uniform(1, 15, 4)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def d_LV():
        """food chain with cycle"""
        growth_rates = np.array([1.0] * 8 + [-0.001] * 4)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Y on X
        A[group_indices['Z'], group_indices['X']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # X on Z

        A[group_indices['Z'], group_indices['X']] += -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['Y'], group_indices['X']] += -0.025 + np.random.normal(0, var, (group_size, group_size))  # Y on X
        A[group_indices['X'], group_indices['Z']] += -0.025 + np.random.normal(0, var, (group_size, group_size))  # X on Z

        group_means = [0.25, 0.25, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1, 55, 4),
            np.random.uniform(1, 30, 4),
            np.random.uniform(1, 15, 4)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def f_wrapped(t, n):
        return f(n, t, growth_rates, A)

    if model_func == "a":
        growth_rates, A, initial_conditions = a_LV()
    elif model_func == "b":
        growth_rates, A, initial_conditions = b_LV()
    elif model_func == "c":
        growth_rates, A, initial_conditions = c_LV()
    elif model_func == "d":
        growth_rates, A, initial_conditions = d_LV()
    elif model_func == "e":
        growth_rates, A, initial_conditions = e_LV()
    elif model_func == "f":
        growth_rates, A, initial_conditions = f_LV()
    else:
        print("Invalid model")

    # Solve stochastic ode but ensure that populations don't become non-negative
    tspan = np.linspace(0, T, int(T * 100))

    solution = solve_ivp(
        f_wrapped,
        [tspan[0], tspan[-1]],
        initial_conditions,
        t_eval=tspan,
        method='RK45'
    )

    solutions = solution.y.T

    # enforce lower bound
    solutions = np.where(solutions < 0.001, 0, solutions)

    # scale up populations
    solutions = solutions * N

    plot_solutions(solutions, tspan, model_func, iter)
    df = create_df(solutions)

    if var == 0.05:
        #plot_solutions(solutions, solution.t, model_func)
        df.to_csv(f'../../data/LV_{model_func}_regression_library.csv', index=False)

    return df

def three_groups_LV_clustered(model_func="food_web", T=50, var=0.0):
    S = 36
    N = 300

    noise_term = 0.0

    # Initialize matrix
    A = np.zeros((S, S))

    # Group boundaries
    group_size = S // 3
    group_indices = {
        'X': slice(0, group_size),
        'Y': slice(group_size, 2 * group_size),
        'Z': slice(2 * group_size, S)
    }

    def a_LV():
        """
        Constant interaction network
        """
        growth_rates = np.ones(S)

        # Equal competition for all species
        A = 0.125 * np.ones((S, S)) + np.random.normal(0, var, (S, S))

        # Intra-group competition: still per-individual (diagonal within group block)
        group_means = [0.25, 0.25, 0.25]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = (np.random.uniform
                (0,100,S))
        initial_conditions /= np.sum(initial_conditions)
        return growth_rates, A, initial_conditions

    def e_LV():
        """
        Food chain
        """
        growth_rates = np.array([1]*12 + [0.2]*12 + [-0.001]*12)

        # Inter-group competition with per-individual variability
        A[group_indices['Y'], group_indices['Z']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Z on Y
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Y on X

        A[group_indices['Z'], group_indices['Y']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Y on Z
        A[group_indices['Y'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # X on Y

        group_means = [0.25, 0.025, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1,55,12),
            np.random.uniform(1,30, 12),
            np.random.uniform(1, 15, 12)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def b_LV():
        """
        Two predators one prey
        """
        growth_rates = np.array([1.0]*12 + [-0.001]*24)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Z on X
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Y on X

        A[group_indices['X'], group_indices['Z']] = -0.025 + np.random.normal(0, var,(group_size, group_size)) # Z on X
        A[group_indices['X'], group_indices['Y']] = -0.025 + np.random.normal(0, var,(group_size, group_size)) # Y on X

        group_means = [0.25, 0.025, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1,65, 12),
            np.random.normal(1, 35, 24)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def c_LV():
        """
        One predator two prey
        """
        growth_rates = np.array([1.0] * 24 + [-0.001] * 12)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Y'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on Y

        A[group_indices['Z'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Z'], group_indices['Y']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on Y

        group_means = [0.25, 0.25, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1,65, 24),
            np.random.uniform(1, 35, 12)
        ])
        initial_conditions = np.clip(initial_conditions, 0.01, None)
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def f_LV():
        """
        Food chain with omnivory
        """
        growth_rates = np.array([1.0] * 12 + [0.2] * 12 + [-0.001] * 12)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Y'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Y on X

        A[group_indices['Z'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Z'], group_indices['Y']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['Y'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Y on X

        group_means = [0.25, 0.025, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1, 55, 12),
            np.random.uniform(1, 30, 12),
            np.random.uniform(1, 15, 12)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def d_LV():
        """food chain with cycle"""
        growth_rates = np.array([1.0] * 8 + [-0.001] * 4)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Y on X
        A[group_indices['Z'], group_indices['X']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # X on Z

        A[group_indices['Z'], group_indices['X']] += -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['Y'], group_indices['X']] += -0.025 + np.random.normal(0, var, (group_size, group_size))  # Y on X
        A[group_indices['X'], group_indices['Z']] += -0.025 + np.random.normal(0, var, (group_size, group_size))  # X on Z

        group_means = [0.25, 0.25, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1, 55, 4),
            np.random.uniform(1, 30, 4),
            np.random.uniform(1, 15, 4)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def f_wrapped(t, n):
        return f(n, t, growth_rates, A)

    if model_func == "a":
        growth_rates, A, initial_conditions = a_LV()
    elif model_func == "b":
        growth_rates, A, initial_conditions = b_LV()
    elif model_func == "c":
        growth_rates, A, initial_conditions = c_LV()
    elif model_func == "d":
        growth_rates, A, initial_conditions = d_LV()
    elif model_func == "e":
        growth_rates, A, initial_conditions = e_LV()
    elif model_func == "f":
        growth_rates, A, initial_conditions = f_LV()
    else:
        print("Invalid model")

    # Solve stochastic ode but ensure that populations don't become non-negative
    tspan = np.linspace(0, T, int(T * 100))

    solution = solve_ivp(
        f_wrapped,
        [tspan[0], tspan[-1]],
        initial_conditions,
        t_eval=tspan,
        method='RK45'
    )

    solutions = solution.y.T

    # enforce lower bound
    solutions = np.where(solutions < 0.001, 0, solutions)

    # scale up populations
    solutions = solutions * N

    #plot_solutions(solutions, tspan, model_func)
    df = create_df_clustered(solutions)

    if var == 0.05:
        plot_solutions(solutions, solution.t, model_func)
        df.to_csv(f'../../data/LV_{model_func}_regression_library.csv', index=False)

    return df


def do_polynomial_regression(df, LV_model, var, regression_type='global', cluster=""):
    model = LinearRegression()

    # Separate target and features
    y = df['dn']
    X = df.drop(columns='dn')

    # Compute polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)

    try:
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
    except ValueError:
        print("Polynomial features failed")
        return y, None

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # Fit model
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    # De-standardize coefficients
    beta_std = model.coef_
    mu = scaler.mean_
    sigma = scaler.scale_

    beta_orig = beta_std / sigma
    intercept_orig = model.intercept_ - np.sum((beta_std * mu) / sigma)

    # Combine into DataFrame
    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': beta_orig
    })

    # Add intercept as a separate row (optional but useful)
    coeff_df.loc[len(coeff_df)] = ['Intercept', intercept_orig]

    # Save
    if var == 0.05:
        coeff_df.to_csv(
            f'METimE_{LV_model}_dn_{regression_type}{cluster}.csv',
            index=False
        )

    return y, y_pred, coeff_df


def set_up_regression(df, var, N_clusters=None, LV_model='constant', regression_type="global", cluster=""):
    # Single observation interval
    all_census = sorted(df['census'].unique())
    reduced_census = deepcopy(all_census)

    # Filter to current censuses
    df_filtered = df[df['census'].isin(reduced_census)].copy().reset_index(drop=True)

    # Recompute dn, dN, dS
    df_deltas = compute_deltas(df_filtered, 'LV').reset_index(drop=True)
    df_deltas = compute_deltas(df_filtered, 'LV').reset_index(drop=True)

    if regression_type=="clustered":
        df_deltas = df_deltas.merge(N_clusters, on='census', how='left')

    # METimE regression
    cols_to_exclude = ['dN', 'n_next', 'N_next', 'S_next', 'dS', 'census', 'species', 'S_t']
    df_for_setup = df_deltas.drop(columns=cols_to_exclude)
    # X, y, census, species = polynomial_regression.set_up_library(df_for_setup, 3, False, False, False)
    y, y_pred, coeffs = do_polynomial_regression(df_for_setup, LV_model, var, regression_type, cluster)

    return y, y_pred, coeffs



if __name__ == "__main__":

    for model in ['c']:
        for i in range(20):
            three_groups_LV(model, T=20, var=0.0, iter=i)

    from PIL import Image
    import os

    # Path where all your figures are stored
    base_path = r""

    # Models
    models = ['a', 'b', 'c', 'd', 'e', 'f']

    # Number of iterations per model
    n_iter = 20

    for model in models:
        print(f"Processing model {model}...")

        # List to hold images
        imgs = []

        # Load all 20 images for this model
        for i in range(n_iter):
            file_name = f"dynamics_{model}_{i}.png"
            file_path = os.path.join(base_path, file_name)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found!")
                continue
            img = Image.open(file_path).convert("RGBA")
            imgs.append(img)

        if not imgs:
            print(f"No images found for model {model}, skipping...")
            continue

        # Initialize base canvas (fully transparent)
        width, height = imgs[0].size
        base = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Overlay all images
        for img in imgs:
            base = Image.alpha_composite(base, img)

        # Save overlay image
        out_path = os.path.join(base_path, f"dynamics_{model}_overlay.png")
        base.save(out_path)
        print(f"Saved overlay for model {model} → {out_path}")
