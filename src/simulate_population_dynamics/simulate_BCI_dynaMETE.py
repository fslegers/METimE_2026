import csv
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from src.MaxEnt_inference.METimE import make_initial_guess, partition_function, ecosystem_structure_function
from src.MaxEnt_inference.METimE import run_optimization as METE
from src.parametrize_transition_functions.dynamete_transition_functions import get_functions, get_function_values, get_transition_function_accuracy
from src.MaxEnt_inference.METimE import run_optimization as METimE
from src.MaxEnt_inference.empirical_BCI import check_constraints, evaluate_model
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def partition_function_given_n(X, n, lambdas):
    return (np.exp(-lambdas[0] * n) - np.exp(-lambdas[0] * n - X['E'] * lambdas[1] * n)) / (lambdas[1] * n)


def get_SAD(lambdas, X, de):
    # """ Calculates species abundance distribution from Lagrange multipliers """
    functions = get_functions()[:2]
    func_vals, _ = get_function_values(functions, X, maxima = [X['N_t'], 1, X['E_t']])
    Z = partition_function(lambdas, func_vals)
    R = ecosystem_structure_function(lambdas, func_vals, Z)
    sad = np.sum(R, axis=1)
    return sad


def cum_SAD(lambdas, X, de):
    sad = get_SAD(lambdas, X, de)
    cum_sad = []

    p = 0
    for n in range(int(X['N_t'])):
        p += sad[n]
        cum_sad.append(p)

    return cum_sad


def sample_community(X):
    temp_X = {'S_t': int(X['S']), 'N_t': int(X['N']), 'E_t': float(X['E'])}

    # Precompute functions(n, e)
    de = 0.5
    functions = get_functions()[:2]
    func_vals, _ = get_function_values(functions, temp_X, maxima=[X['N'], 1, X['E']])

    initial_lambdas = make_initial_guess(temp_X)
    lambdas = METE(
        initial_lambdas[:2],
        {
            'N/S': float(X['N'] / X['S']),
            'E/S': float(X['E'] / X['S'])
        },
        temp_X,
        func_vals,
        de,
        optimizer='trust-constr'
        #maxiter=15
    )

    # Sample species
    p = np.random.uniform(0, 1, X['S'])
    cum_sad = cum_SAD(lambdas, temp_X, de)

    populations = []
    for prob in p:
        n = int(np.searchsorted(cum_sad, prob) + 1)
        populations.append(n)
    species_indices = np.cumsum(populations)
    species_indices = np.concatenate(([0], species_indices))

    # Sample individual metabolic rates
    individuals = []
    for pop in populations:
        Z_n = partition_function_given_n(X, pop, lambdas)
        CDF_inverse = lambda u: -(np.log(1 - Z_n * lambdas[1] * n * u * np.exp(lambdas[0] * n)))/(lambdas[1] * n)
        u_samples = np.random.uniform(0, 1, pop)
        samples = [float(CDF_inverse(u).real) for u in u_samples] # checked: there is no imaginary part
        while np.isnan(samples).any():
            u_samples = np.random.uniform(0, min(1, 1/(Z_n * lambdas[1] * n * np.exp(lambdas[0] * n))), pop)
            samples = [float(CDF_inverse(u).real) for u in u_samples]
        individuals += samples
    tree_id_list = list(range(len(individuals)))

    return individuals, species_indices, tree_id_list


def update_metabolic_rates(e, X, dt, param):
    """ Euler method to update metabolic rates """
    de_dt = np.maximum(0, param['w'] * e ** (2 / 3) - param['w1'] * e)
    new_e = e + dt * de_dt
    X['E'] = np.sum(new_e)
    return new_e, X


def update_event_rates(species_ids, abundances, metabolic_rates, X, p):
    """ Compute birth and death rates """
    n = np.array([abundances[sp] for sp in species_ids])
    birth_rates = p['b'] * n * (metabolic_rates ** (-1/3))
    death_rates = (p['d0'] + p['d1'] * n + p['d'] * (X['E'] / p['Ec'])) * n * (metabolic_rates ** (-1/3))
    migration_rates = p['m'] * np.array(list(abundances.values())) / X['N']
    R = birth_rates.sum() + death_rates.sum() + migration_rates.sum()
    return birth_rates, death_rates, migration_rates, R


def what_event_happened(birth_rates, death_rates, migration_rates, R, q):
    event_rates = np.concatenate([birth_rates, death_rates, migration_rates])
    cumulative_rates = np.cumsum(event_rates)

    index = np.searchsorted(cumulative_rates, q * R)

    if index < len(birth_rates):
        return ('birth', index)
    elif index < len(birth_rates) + len(death_rates):
        return ('death', index - len(birth_rates))
    else:
        return ('migration', index - len(birth_rates) - len(death_rates))


def perform_event(tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X, event_info, params):
    event_type, idx = event_info

    if (event_type == 'birth'):
        tree_ids = np.append(tree_ids, next_tree_id)
        species_ids = np.append(species_ids, species_ids[idx])
        metabolic_rates = np.append(metabolic_rates, 1.0)
        abundances[species_ids[idx]] += 1
        next_tree_id += 1
        X['N'] += 1
        X['E'] += 1

    elif event_type == 'death':
        tree_ids = np.delete(tree_ids, idx)
        abundances[species_ids[idx]] -= 1
        if abundances[species_ids[idx]] == 0:
            del abundances[species_ids[idx]]
            X['S'] -= 1
        species_ids = np.delete(species_ids, idx)
        X['E'] -= metabolic_rates[idx]
        X['N'] -= 1
        metabolic_rates = np.delete(metabolic_rates, idx)

    else: # Migration
        prob_new_species = np.exp(-params['mu_meta'] * X['S'] - np.euler_gamma)
        if np.random.rand() < prob_new_species or len(tree_ids) == 0:
            # Create a new species
            new_species_id = max(species_ids) + 1 if len(species_ids) > 0 else 0
            tree_ids = np.append(tree_ids, next_tree_id)
            species_ids = np.append(species_ids, new_species_id)
            metabolic_rates = np.append(metabolic_rates, 1.0)
            abundances[new_species_id] = 1
            X['N'] += 1
            X['S'] += 1
            X['E'] += 1.0
            next_tree_id += 1
        else:
            # Add to existing species (species_ids[idx])
            tree_ids = np.append(tree_ids, next_tree_id)
            species_ids = np.append(species_ids, species_ids[idx])
            metabolic_rates = np.append(metabolic_rates, 1.0)
            abundances[species_ids[idx]] += 1
            X['N'] += 1
            X['E'] += 1.0
            next_tree_id += 1

    return tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X


def gillespie(metabolic_rates, species_indices, tree_id_list, p, t_max, obs_interval, save_final_state=False):
    # some preparations
    species_ids = np.zeros(int(species_indices[-1]), dtype=int)
    for i in range(1, len(species_indices)):
        start = species_indices[i - 1]
        end = species_indices[i]
        species_ids[start:end] = i

    observation_times = np.arange(0, t_max, obs_interval)
    obs_pointer = 0

    births = 0
    deaths = 0
    migrations = 0

    # Initialization
    tree_ids = np.array(tree_id_list)
    species_ids = np.array(species_ids)
    metabolic_rates = np.array(metabolic_rates)
    abundances = dict(Counter(species_ids))

    try:
        next_tree_id = tree_ids.max() + 1
    except:
        next_tree_id = 0

    # Compute state variables and event rates
    X = {'S':len(np.unique(species_ids)), 'N': len(np.unique(tree_ids)), 'E': sum(metabolic_rates)}
    birth_rates, death_rates, migration_rates, R = update_event_rates(species_ids, abundances, metabolic_rates, X, p)

    # Store snapshots here
    snapshots = []

    # Start simulation
    t = 0
    n_iter = 0

    print(f"N: {len(metabolic_rates)}")

    # initialize tqdm bar
    pbar = tqdm(total=t_max, desc="Simulating", unit="time")

    while t < t_max:
        # Sample event time
        if len(tree_id_list) == 0:
            time_until_event = 0
        else:
            u = np.random.uniform(0, 1)
            time_until_event = -np.log(u) / R
        t += time_until_event
        n_iter += 1

        # update progress bar (clamp to not overshoot t_max)
        pbar.update(min(time_until_event, t_max - pbar.n))

        # In case the event happens *after* the current observation time, save a snapshot of the community
        while obs_pointer < len(observation_times) and t > observation_times[obs_pointer]:
            # Save snapshot
            snapshots.append({
                't': observation_times[obs_pointer],
                'species_ids': species_ids,
                'tree_ids': tree_ids,
                'S': X['S'],
                'N': X['N'],
                'E': X['E'],
                'n': np.array([abundances[sp] for sp in species_ids]),
                'e': metabolic_rates.copy()
            })

            # Progress observation time
            obs_pointer += 1

        # Update individual metabolic rates
        metabolic_rates, X = update_metabolic_rates(metabolic_rates, X, time_until_event, p)

        # Update event rates based on new metabolic rates e
        birth_rates, death_rates, migration_rates, R = update_event_rates(species_ids, abundances, metabolic_rates, X, p)

        # Determine what event (birth or death) happened
        q = np.random.uniform(0, 1)
        event = what_event_happened(birth_rates, death_rates, migration_rates, R, q)

        if event[0] == "birth":
            births += 1
        elif event[0] == "death":
            deaths += 1
        else:
            migrations += 1

        tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X = perform_event(tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X, event, p)
    pbar.close()

    # Flatten list of snapshowts with one row per individual per time step
    rows = []
    for snap in snapshots:
        n_individuals = len(snap['e'])
        for i in range(n_individuals):
            rows.append({
                't': snap['t'],
                'species_id': snap['species_ids'][i],
                'tree_id': snap['tree_ids'][i],
                'S': snap['S'],
                'N': snap['N'],
                'E': snap['E'],
                'n': snap['n'][i],
                'e': snap['e'][i]
            })
    df = pd.DataFrame(rows)

    if save_final_state:
        output_file = "simulated_dynaMETE_snapshots.csv"
        df.to_csv(output_file, index=False)

        # Save final community state to resume later
        final_state_file = "final_community_state.npz"

        # Reconstruct species_indices from species_ids
        unique_species, counts = np.unique(species_ids, return_counts=True)
        species_indices_final = np.cumsum(np.insert(counts, 0, 0))  # like original format

        np.savez(
            final_state_file,
            metabolic_rates=metabolic_rates,
            species_indices=species_indices_final,
            tree_id_list=tree_ids,
            X=np.array([X['S'], X['N'], X['E']])
        )

        # Save param dictionary as JSON
        param_file = final_state_file.replace(".npz", "_param.json")
        with open(param_file, 'w') as f:
            json.dump(p, f)

    return df


def plot_state_var(df, frac):
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })


    # Create three horizontally aligned subplots
    fig, axes = plt.subplots(ncols=3, figsize=(18, 5), sharex=True)

    # Plot S vs time
    axes[0].plot(df['t'], df['S'], color='tab:blue', marker='o')
    axes[0].set_xlabel('time (t)')
    axes[0].set_ylabel(r'$S_t$')
    axes[0].grid(True)

    # Plot N vs time
    axes[1].plot(df['t'], df['N'], color='tab:green', marker='s')
    axes[1].set_xlabel('time (t)')
    axes[1].set_ylabel(r'$N_t$')
    axes[1].grid(True)

    # Plot E vs time
    axes[2].plot(df['t'], df['E'], color='tab:red', marker='^')
    axes[2].set_xlabel('time (t)')
    axes[2].set_ylabel(r'$E_t$')
    axes[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.savefig('equilibrium', dpi=300, transparent=True)
    plt.show()


def get_empirical_RAD(df, census):
    df = df[df['census'] == census]
    df = df[['species', 'n']].drop_duplicates()

    # Create rank abundance distribution
    df = df.sort_values(by='n', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    rad = df[['rank', 'n']].rename(columns={'n': 'abundance'})

    return rad


def remove_fraction(frac, metabolic_rates, species_indices, tree_id_list):
    # Remove fraction of community
    # indices to remove
    indices_to_remove = np.random.choice(len(tree_id_list), int(frac * len(tree_id_list)), replace=False)

    # remove from tree_id_list and metabolic_rates
    # create species_id from species_indices
    species_id = np.empty(len(tree_id_list), dtype=int)
    for i, start_idx in enumerate(species_indices):
        end_idx = species_indices[i + 1] if i + 1 < len(species_indices) else len(tree_id_list)
        species_id[start_idx:end_idx] = i

    tree_id_list = np.array([tree for i, tree in enumerate(tree_id_list) if i not in indices_to_remove])
    metabolic_rates = np.array([rate for i, rate in enumerate(metabolic_rates) if i not in indices_to_remove])
    species_id = np.array([species for i, species in enumerate(species_id) if i not in indices_to_remove])

    unique_species, counts = np.unique(species_id, return_counts=True)
    species_indices = np.concatenate(([0], np.cumsum(counts)))

    # recalculate X
    S = len(species_indices) - 1
    N = len(tree_id_list)
    E = metabolic_rates.sum()
    X = {'S': S, 'N': N, 'E': E}

    return metabolic_rates, species_indices, tree_id_list, X


def run_simulation(X, p, frac, n_iter=1, t_max=30.0, obs_interval=0.25, start_from_prev=False, save_final_state=False):
    if start_from_prev:
        final_state_file = "final_community_state.npz"
        data = np.load(final_state_file)

        metabolic_rates = data['metabolic_rates']
        species_indices = data['species_indices']
        tree_id_list = data['tree_id_list']
        X_vals = data['X']
        X = {'S': int(X_vals[0]), 'N': int(X_vals[1]), 'E': float(X_vals[2])}

    else:
        metabolic_rates, species_indices, tree_id_list = sample_community(X)
        X['S'], X['N'], X['E'] = len(species_indices) - 1, len(metabolic_rates), sum(metabolic_rates)

    metabolic_rates, species_indices, tree_id_list, X = remove_fraction(frac, metabolic_rates, species_indices, tree_id_list)

    results_list = []

    for iter in range(n_iter):
        # Generate trajectories of its state variables
        df = gillespie(metabolic_rates, species_indices, tree_id_list, p, t_max=t_max, obs_interval=obs_interval, save_final_state=save_final_state)

        if iter == 0:
            plot_state_var(df, frac)

        # Rescale t
        df['t'] = (df['t'] * (1 / obs_interval)).astype(int)

        # Add dn and de
        df = df.sort_values(by=['tree_id', 't'])
        df['dn'] = df.groupby('tree_id')['n'].shift(-1) - df['n']
        df['de'] = df.groupby('tree_id')['e'].shift(-1) - df['e']

        # For the last row of each group, shift(-1) is NaN → replace with -n, -e
        last_in_group = ~df['tree_id'].eq(df['tree_id'].shift(-1))
        df.loc[last_in_group, 'dn'] = -df.loc[last_in_group, 'n']
        df.loc[last_in_group, 'de'] = -df.loc[last_in_group, 'e']

        global_ts = (
            df.drop_duplicates("t")[["t", "N", "E", "S"]]  # keep one row per t
            .sort_values("t")
        )

        # compute differences
        global_ts["dN/S"] = (global_ts["N"].shift(-1) - global_ts["N"]) / global_ts["S"]
        global_ts["dE/S"] = (global_ts["E"].shift(-1) - global_ts["E"]) / global_ts["S"]

        # merge back into full df
        df = df.merge(global_ts[["t", "dN/S", "dE/S"]], on="t", how="left")

        df = df.dropna()

        # Rename columns
        df = df.rename(columns={
            'S': 'S_t',
            'N': 'N_t',
            'E': 'E_t',
            't': 'census',
            'tree_id': 'TreeID',
            'species_id': 'species'
        })

        # Compute polynomial coefficients
        make_boxplot(df, frac)
        r2_dn, r2_de = get_transition_function_accuracy(df)
        functions = get_functions()

        # For 3 cencuses, perform MaxEnt inference
        for census in [1, 5, 10]:
            input_census = df[df['census'] == census]

            X = input_census[[
                'S_t', 'N_t', 'E_t',
            ]].drop_duplicates().iloc[0]

            macro_var = {
                'N/S': float(X['N_t'] / X['S_t']),
                'E/S': float(X['E_t'] / X['S_t']),
                'dN/S': input_census['dN/S'].unique()[0],
                'dE/S': input_census['dE/S'].unique()[0]
            }

            print("macro_var")
            print(macro_var)

            # Get empirical rank abundance distribution
            empirical_rad = get_empirical_RAD(input_census, census)['abundance']

            # Precompute functions(n, e)
            max_n = int(X['N_t'] - X['S_t'])
            min_e = 1
            max_e = min(X['E_t'], max(input_census['e']))

            func_vals, _ = get_function_values(functions, X,
                                               [max_n, min_e, max_e],)

            #######################################
            #####            METE             #####
            #######################################
            print(" ")
            print("----------METE----------")
            # Make initial guess
            initial_lambdas = make_initial_guess(X)
            METE_lambdas = METE(
                initial_lambdas[:2],
                {
                    'N/S': float(X['N_t'] / X['S_t']),
                    'E/S': float(X['E_t'] / X['S_t'])
                },
                X,
                func_vals[:2],
                optimizer='trust-constr',
                maxiter=5e5
            )
            METE_lambdas = np.append(METE_lambdas, [0, 0])
            print("Optimized lambdas: {}".format(METE_lambdas[:2]))
            mete_constraint_errors = check_constraints(METE_lambdas, input_census, func_vals)
            METE_results, METE_rad = evaluate_model(METE_lambdas, X, func_vals, empirical_rad, mete_constraint_errors)
            #METE_lambdas = np.append(METE_lambdas, [0, 0])
            print(
                f"AIC: {METE_results['AIC'].values[0]}, RMSE: {METE_results['RMSE'].values[0]}, MAE: {METE_results['MAE'].values[0]}")

            prev_best_MAE = np.inf
            for w_exp in np.arange(-2, 2, 1, dtype=float):
                for w_base in [1, 3, 5, 7, 9]:
                    w = w_base * 10 ** w_exp
                    print("Slack weight = {w}".format(w=w))

                    #######################################
                    #####           METimE            #####
                    #######################################
                    print(" ")
                    print("----------METimE----------")
                    METimE_lambdas = METimE(
                        METE_lambdas[:4],
                        {
                            'N/S': float(X['N_t'] / X['S_t']),
                            'E/S': float(X['E_t'] / X['S_t']),
                            'dN/S': input_census['dN/S'].unique()[0],
                        },
                        func_vals[:4],
                        w,
                        maxiter=5e5
                    )
                    print("Optimized lambdas: {}".format(METimE_lambdas[:3]))

                    METimE_lambdas = np.append(METimE_lambdas[:4], [0])
                    metime_constraint_errors = check_constraints(METimE_lambdas, input_census, func_vals)
                    METimE_results, METimE_rad = evaluate_model(METimE_lambdas, X, func_vals, empirical_rad, metime_constraint_errors)
                    print(f"AIC: {METimE_results['AIC'].values[0]}, RMSE: {METimE_results['RMSE'].values[0]}, MAE: {METimE_results['MAE'].values[0]}")

                    ##########################################
                    #####           Save results         #####
                    ##########################################
                    results_list.append({
                        'iter': iter,
                        'frac': frac,
                        'census': census,
                        'slack_weight': w,
                        'r2_dn': r2_dn,
                        'r2_de': r2_de,
                        'METE_error_N/S': mete_constraint_errors[0],
                        'METE_error_E/S': mete_constraint_errors[1],
                        'METE_error_dN/S': mete_constraint_errors[2],
                        'METE_error_dE/S': mete_constraint_errors[3],
                        'METimE_error_N/S': metime_constraint_errors[0],
                        'METimE_error_E/S': metime_constraint_errors[1],
                        'METimE_error_dN/S': metime_constraint_errors[2],
                        'METimE_error_dE/S': metime_constraint_errors[3],
                        'METE_AIC': METE_results['AIC'].values[0],
                        'METE_MAE': METE_results['MAE'].values[0],
                        'METE_RMSE': METE_results['RMSE'].values[0],
                        'METimE_AIC': METimE_results['AIC'].values[0],
                        'METimE_MAE': METimE_results['MAE'].values[0],
                        'METimE_RMSE': METimE_results['RMSE'].values[0]
                    })

                    # add_row({
                    #     "frac": frac,
                    #     "census": census,
                    #     "slack_weight": w,
                    #     "AIC": METimE_results['AIC'].values[0],
                    #     "RMSE": METimE_results['RMSE'].values[0],
                    #     "MAE": METimE_results['MAE'].values[0],
                    #     "entropy": -entropy(METimE_lambdas[:4], func_vals)
                    # })

                    if METimE_results['MAE'].values[0] < prev_best_MAE:
                        ext = f"dynamete_iter={iter}_frac={frac}.png"
                        #plot_RADs(empirical_rad, METE_rad, METimE_rad, ext, obs_label="Simulated", weight=w, use_log=True)
                        prev_best_MAE = METimE_results['MAE'].values[0]

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'dynamete_{frac}.csv', index=False)

def make_boxplot(df, frac=0.0):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))  # 1 row, 2 columns
    fig.suptitle(f"Fraction removed: {frac}", fontsize=16)  # figure title

    # Left subplot: boxplot of df['dn']
    sns.boxplot(y=df["dn"], ax=axes[0], showfliers=False)
    axes[0].set_title("dn")  # optional axis title
    axes[0].set_xlabel("")  # remove x-label if you like
    #axes[0].set_yscale("symlog")

    # Right subplot: boxplot of df['de']
    sns.boxplot(y=df["de"], ax=axes[1], showfliers=False)
    axes[1].set_title("de")
    axes[1].set_xlabel("")
    #axes[1].set_yscale("symlog")

    # Make sure subplots do not share y scale
    for ax in axes:
        ax.autoscale(enable=True, axis='y', tight=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()

def load_simple_dynaMETE():
    # lowered migration rate from 436.3 to 250
    # param = {
    #     'b': 0.2, 'd': 0.2, 'Ec': 450, 'm': 436.3,
    #     'w': 1.0, 'w1': 0.42, 'mu_meta': 0.0215,
    #     'd0': 0,
    #     'd1': 0
    # }
    param = {
            'b': 0.5, 'd': 0.15504447, 'Ec': 1700, 'm': 1000,
            'w': 1.0, 'w1': 0.083626, 'mu_meta': 0.0269564,
            'd0': 0.1,
            'd1': 0.0000324
    }

    # X = {
    #     'E': 700,
    #     'N': 340,
    #     'S': 130,
    #     'beta': 0.0001
    # }
    X = {
        'E': 3500,
        'N': 3500,
        'S': 130,
        'beta': 0.0001
    }
    return param, X


def load_new_dynaMETE():
    param = {
            'b': 0.5, 'd': 0.15504447, 'Ec': 1e07, 'm': 1000,                                                           \
            'w': 1.0, 'w1': 0.083626, 'mu_meta': 0.0269564,
            'd0': 0.1,
            'd1': 0.0000324
    }

    X = {
            'E': 2e07,
            'N': 230000,
            'S': 320,
            'beta': 0.0001
    }

    return param, X


def add_row(data):
    filename = "results_per_slack_weight.csv"
    file_exists = os.path.isfile(filename)
    columns = ["frac", "census", "slack_weight", "AIC", "RMSE", "MAE", "entropy"]

    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(data)


if __name__ == '__main__':
    random.seed(123)

    param, X = load_simple_dynaMETE()
    #_, _, _ = run_simulation(X, param, frac=0, t_max=10.0, obs_interval=0.5, save_final_state=True, start_from_prev=False) # Only needed once to get to an equilibrium

    # Then, repeatedly run simulation from equilibrium, where different fractions of the initial population are removed
    # thereby disturbing the equilibrium
    # for each "level of disturbance" (fraction of initial population removed) repeat 5 times

    for frac in [0.0, 0.2, 0.4, 0.6, 0.8]:
        print(f"-----------Running simulation for frac={frac}----------------")
        run_simulation(X, param, frac, n_iter=20, t_max=0.05, obs_interval=0.01, start_from_prev=True)



