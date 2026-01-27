import os
from collections import defaultdict
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rv_discrete
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from src.parametrize_transition_functions.SINDy_like_regression_for_LV import do_polynomial_regression as sindy
from src.MaxEnt_inference.METimE import run_optimization as METE
from src.parametrize_transition_functions.SINDy_like_regression_for_LV import get_functions, get_function_values
from src.simulate_population_dynamics.simulate_LV import three_groups_LV_clustered

import warnings
warnings.filterwarnings("ignore")

def get_rank_abundance(sad, X):
    """
    Generate a predicted rank-abundance distribution using the quantile method.
    Ensures exactly S_t values by clipping quantiles and handling edge cases.
    """
    S = int(X['S_t']) + 1

    if np.sum(sad) > 0:
        sad = sad / np.sum(sad)
    else:
        sad = np.ones_like(sad) / len(sad)

    # Create the discrete distribution
    n_vals = np.arange(1, len(sad) + 1)
    dist = rv_discrete(name='sad_dist', values=(n_vals, sad))

    # Safer quantiles: strictly within (0, 1)
    epsilon = 1e-6
    quantiles = (np.arange(1, S + 1) - 0.5) / S
    quantiles = np.clip(quantiles, epsilon, 1 - epsilon)

    # Evaluate quantiles
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_abundances = dist.ppf(quantiles).astype(int)

    # Fix any zeros or nans (can happen if ppf fails)
    pred_abundances = np.where(pred_abundances < 1, 1, pred_abundances)
    pred_abundances = np.nan_to_num(pred_abundances, nan=1).astype(int)

    # Ensure output length = S_t
    if len(pred_abundances) != S:
        raise ValueError(f"Expected {S} predicted abundances, got {len(pred_abundances)}.")

    return np.sort(pred_abundances)[::-1]  # descending order

def partition_function(lambdas, func_vals):
    lambdas = np.asarray(lambdas).reshape(-1, 1, 1)
    exponent_matrix = np.sum(-lambdas * func_vals, axis=0)
    Z = np.exp(exponent_matrix).sum()

    if np.isclose(Z, 0.0, atol=1e-12):
        print("Invalid values detected in Z")
        #Z = 1e-12

    if np.isinf(Z):
        Z = 1e10

    return Z

def ecosystem_structure_function(lambdas, func_vals, Z):
    lambdas_arr = np.asarray(lambdas).reshape(-1, 1, 1)
    exponent_matrix = np.sum(-lambdas_arr * func_vals, axis=0)
    R = np.exp(exponent_matrix) / Z

    if np.isnan(R).any():
        raise ValueError("NaN values found in R — check Z and exponent_matrix ranges.")

    return R

def entropy(vars, func_vals, scales=[1,1,1,1,1,1]):
    """
    Computes Shannon entropy: -sum R(n,e) * log(R(n,e))
    """
    # Scale back lambdas
    lambdas = vars[:4] * scales[:4]

    # Partition function Z
    Z = partition_function(lambdas, func_vals)

    # Ecosystem structure function R
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    # Negative shannon entropy (because we minimize instead of maximize)
    H = np.sum(np.where(R > 0, R * np.log(R), 0)) # Only compute log(R) where R > 0 and put 0 otherwise

    # Safety check
    if np.isnan(H) or np.isinf(H):
        print("Invalid values detected in entropy")
        H = 1e10

    return H

def get_relative_errors(vars, func_vals, func_index, target_value, scales):
    """
    Vectorized single constraint calculation
    Weight determines the relative weight of the constraint
    """
    # Scale back lambdas
    vars = vars * scales
    lambdas = vars[:2]

    # Partition function Z
    Z = partition_function(lambdas, func_vals)

    # Ecosystem structure function R
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    # Expected value of f_k under R
    expected_value = np.sum(R * func_vals[func_index])

    # Safety check
    if np.isnan(expected_value) or np.isinf(expected_value):
        print("Invalid values detected in single constraint")
        expected_value = 1e10

    # Return deviation from target
    return ((expected_value - target_value) / target_value)

def penalized_entropy(vars, func_vals, macro_var, scales, slack_weight = 1.0):
    return (entropy(vars, func_vals, scales) + slack_weight * (
            (get_relative_errors(vars, func_vals, 1, macro_var['dN/S'], scales)) ** 2))

def beta_function(beta, S, N):
    """
    Beta function used to generate the initial guess for Lagrange multipliers.
    """
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta * (N + 1))) * np.log(1.0 / beta) - S / N

def make_initial_guess(X):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR'diag function meteESF.mete.lambda

    :param state_variables: state variables S, N and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    S, N = int(X['S_t']), int(X['N_t'])

    nom = N * (- np.sqrt( S * (4 * N + S) / N ** 2) + 2 * N + S)
    denom = 2 * N
    l1 = np.log(nom/denom)

    return [l1]

def single_constraint(vars, func_vals, func_index, target_value, scales):
    """
    Vectorized single constraint calculation
    Weight determines the relative weight of the constraint
    """
    # Scale back lambdas
    vars = vars * scales
    lambdas = vars[:4]

    # Partition function Z
    Z = partition_function(lambdas, func_vals)

    # Ecosystem structure function R
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    # Expected value of f_k under R
    expected_value = np.sum(R * func_vals[func_index])

    # Safety check
    if np.isnan(expected_value) or np.isinf(expected_value):
        print("Invalid values detected in single constraint")
        expected_value = 1e10

    # Return deviation from target
    return (expected_value - target_value)

def compute_lambda_bounds(min_f, max_f, max_exp=400):
    eps = 1e-12

    # If function is basically zero, allow wide bounds
    if abs(min_f) < eps and abs(max_f) < eps:
        return (-1, 1)

    candidates = []

    for f_b in [min_f, max_f]:
        if abs(f_b) < eps:
            continue
        lower = -max_exp / f_b
        upper = max_exp / f_b

        # For negative f_b, lower will be > upper; swap them
        if lower > upper:
            lower, upper = upper, lower

        candidates.append((lower, upper))

    # Now intersect bounds: take max of lowers, min of uppers
    lowers, uppers = zip(*candidates)
    lower_bound = max(lowers)
    upper_bound = min(uppers)

    # If bounds are invalid (empty intersection), fallback to some safe range
    if lower_bound > upper_bound:
        return (-1, 1)

    return (lower_bound, upper_bound)

def run_optimization(vars, macro_var, func_vals, slack_weight=1, maxiter=1e08):
    vars = np.asarray(vars, dtype=float)

    # Compute bounds (to prevent overflow in exp)
    f3_vals = func_vals[1, :]
    min_f3, max_f3 = f3_vals.min(), f3_vals.max()
    bounds_dn = compute_lambda_bounds(min_f3, max_f3, 100)

    # Define scale factors so that parameters are roughly of the same order of magnitude
    values = np.asarray([vars[0], bounds_dn[1]], dtype=float)

    scales = np.where(values != 0,
                    10.0 ** np.floor(np.log10(np.abs(values))),
                    1.0)
    vars = vars / scales

    bounds = [
        (0, 18) / scales[0],
        bounds_dn / scales[1]
    ]

    # Collect all constraints
    constraint_order = ['N/S', 'dN/S']

    constraints = [{
        'type': 'eq',
        'fun': lambda vars, F_k=macro_var[name], idx=i:
        single_constraint(vars, func_vals, idx, F_k, scales)
    } for i, name in enumerate(constraint_order)]

    result = minimize(penalized_entropy,
                      vars,
                      args=(func_vals, macro_var, scales, slack_weight),
                      constraints=constraints,
                      bounds=bounds,
                      method="trust-constr",
                      options={'maxiter':maxiter,
                               'initial_tr_radius': 0.05,
                               'initial_constr_penalty': 2.0,
                               'gtol': 1e-12,
                               'disp': True,
                               'verbose': 1})

    optimized_lambdas = result.x * scales

    return optimized_lambdas

def evaluate_model(lambdas, X, func_vals, empirical_rad, constraint_errors):
    Z = partition_function(lambdas[:2], func_vals)
    R = ecosystem_structure_function(lambdas[:2], func_vals, Z)

    # Compute SAD
    sad = np.sum(R, axis=1)
    print("Sum of sad: {}".format(np.sum(sad)))

    while len(sad) < int(X['N_t']):
        sad = np.concatenate([sad, np.zeros(int(X['N_t']) - sad.size)])

    # Resize to match empirical_rad length
    rad = get_rank_abundance(sad, X)
    rad = rad[:len(empirical_rad)]
    empirical_rad = empirical_rad[:len(rad)]

    # MEA
    mae = mean_absolute_error(empirical_rad, rad)

    # RMSE
    rmse = root_mean_squared_error(empirical_rad, rad)

    k = len(lambdas) + 1
    log_likelihood = 0
    for i in range(len(empirical_rad)):
        n_i = int(empirical_rad[i])
        p_i = max(sad[n_i - 1], 1e-8)
        log_likelihood += np.log(p_i)
        log_likelihood += np.log(p_i)
    aic = 2 * k - 2 * log_likelihood

    results_data = {
        'MAE': [mae],
        'AIC': [aic],
        'RMSE': [rmse]
    }

    # Add lambdas to dictionary
    for i, lam in enumerate(lambdas):
        results_data[f'lambda_{i}'] = [lam]

    for constr, error in zip(['N/S', 'E/S', 'dN', 'dE'], constraint_errors):
        results_data[f'{constr}'] = error

    # Create DataFrame
    results_df = pd.DataFrame(results_data)

    return results_df, rad

def check_constraints(lambdas, input, func_vals):
    """
    Returns the error on constraints given some lambda values
    Given in percentage of the observed value
    """
    S, N = (int(input['S_t'].drop_duplicates().iloc[0]),
               int(input['N_t'].drop_duplicates().iloc[0]))

    X = {
        'S_t': S,
        'N_t': N
    }

    macro_var = {
        'N/S': X['N_t'] / X['S_t'],
        'dN/S': input['dN/S'].unique()[0]
    }

    Z = partition_function(lambdas, func_vals)
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    absolute_errors = []
    percentage_errors = []

    constraint_order = ['N/S', 'dN/S']
    for func_index, func_name in enumerate(constraint_order):
        val = macro_var[func_name]
        # Expected value of f_k under R
        expected_value = np.sum(R * func_vals[func_index])

        # Compute constraint error
        abs_error = np.abs(expected_value - val)
        pct_error = abs_error / np.abs(val) * 100

        absolute_errors.append(abs_error)
        percentage_errors.append(pct_error)

    print("\n Errors on constraints:")
    print(f"{'Constraint':<10} {'Abs Error':>15} {'% Error':>15}")
    print("-" * 42)
    for key, abs_err, pct_err in zip(macro_var.keys(), absolute_errors, percentage_errors):
        print(f"{key:<10} {abs_err:15.6f} {pct_err:15.2f}")

    return absolute_errors

def plot_trajectories(df):
    """
    Plots trajectories for variables n, e, S_t, N_t, and E_t as functions of census.

    - n: one line per species
    - e: one line per individual (assuming each row corresponds to an individual, use species + index)
    - S_t, N_t, E_t: one line in total
    """

    # Create a figure with 5 subplots sharing x-axis
    fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)

    # 1. Plot n by species
    sns.lineplot(ax=axes[0], data=df, x='census', y='n', hue='species', marker='o')
    axes[0].set_title('n(t) per species')

    # 2. Plot e by individual (using row index as individual identifier)
    sns.lineplot(ax=axes[1], data=df, x='census', y='e', hue='TreeID', marker='o')
    axes[1].set_title('e(t) per tree')

    # Sort census values
    census_sorted = sorted(df['census'].unique())

    # 3. Plot S_t
    axes[2].plot(
        census_sorted,
        df.groupby('census')['S_t'].mean().reindex(census_sorted),
        marker='o'
    )
    axes[2].set_title('S_t')

    # 4. Plot N_t
    axes[3].plot(
        census_sorted,
        df.groupby('census')['N_t'].mean().reindex(census_sorted),
        marker='o'
    )
    axes[3].set_title('N_t')

    # 5. Plot E_t
    axes[4].plot(
        census_sorted,
        df.groupby('census')['E_t'].mean().reindex(census_sorted),
        marker='o'
    )
    axes[4].set_title('E_t')
    axes[4].set_xlabel('Census')

    plt.tight_layout()
    plt.show()

def combine_rads_per_census(empirical_rad, METE_rad, METimE_rad):
    emp_sorted = sorted(list(empirical_rad), reverse=True)
    mete_sorted = sorted(list(METE_rad), reverse=True)
    metime_sorted = sorted(list(METimE_rad), reverse=True)

    return emp_sorted, mete_sorted, metime_sorted

def evaluate_model_2(empirical_rad, rad):
    mae = mean_absolute_error(empirical_rad, rad)
    rmse = root_mean_squared_error(empirical_rad, rad)

    results_data = {
        'MAE': [mae],
        'RMSE': [rmse]
    }

    results_df = pd.DataFrame(results_data)

    return results_df

def plot_RADs(empirical_rad, METE_rad, METimE_rad, save_name, use_log=False):
    ranks = np.arange(1, len(empirical_rad) + 1)

    # Define custom colors
    redish = "#ef8a62"
    blueish = "#67a9cf"
    greyish = "#4D4D4D"

    plt.rcParams.update({
        'font.size': 20,  # base font size
        'axes.labelsize': 22,  # x and y labels
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 22
    })

    plt.figure(figsize=(6, 4))

    # Plot with updated styles
    plt.plot(ranks, empirical_rad, 'o-', color=greyish, markersize=6, linewidth=2, label='Simulated')
    plt.plot(ranks, METE_rad, 's--', color=blueish, markersize=6, linewidth=2, label='METE')
    plt.plot(ranks, METimE_rad, '^--', color=redish, markersize=6, linewidth=2, label='METimE')

    plt.xlabel("Rank", fontsize=16)

    # Handle log scale
    if use_log:
        plt.yscale("log") # base is 10
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        ylabel = r"$\log_{10}(\mathrm{Abundance})$"
    else:
        ylabel = "Abundance"

    plt.ylabel(ylabel, fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = f'{save_name}.png'
    os.makedirs("results", exist_ok=True)

    if save_name is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":

    for model in ['a', 'b', 'c', 'd', 'e', 'f']:
         for var in [0.05]:

            for iter in range(1): # should be 25
                ext = f'_model={model}_var={var}_iter={iter}'
                df_full = three_groups_LV_clustered(model, T=8, var=var)

                for cluster in [1,2,3]:
                    df = df_full[df_full['cluster'] == cluster]
                    df.drop(columns=['cluster'], inplace=True)

                    # choose only a small number of censuses to do the analysis on
                    censuses = df['census'].unique()[::4]

                    # Compute polynomial coefficients
                    try:
                        alphas, r2_dn, scaler = sindy(df)
                    except:
                        print("error in calculation of polynomial coefficients")
                        sindy(df)
                    functions = get_functions()
                    alphas = alphas['Coefficient'].values

                    # Create list to store results
                    results_list = []

                    per_census_empirical = defaultdict(list)
                    per_census_METE = defaultdict(list)
                    per_census_METimE = defaultdict(list)

                    for census in df['census'].unique():
                        print(f"\n Census: {census} \n")
                        input_census = df[df['census'] == census]

                        X = input_census[[
                            'S_t', 'N_t'
                        ]].drop_duplicates().iloc[0]

                        macro_var = {
                            'N/S': float(X['N_t'] / X['S_t']),
                            'dN/S': input_census['dN/S'].unique()[0]
                        }

                        # Precompute functions(n, e)
                        func_vals = get_function_values(functions, X, alphas, scaler,
                                                        show_landscape=True)

                        # Get empirical rank abundance distribution
                        grouped = input_census.groupby('species')['n'].sum()
                        empirical_rad = grouped.sort_values(ascending=False).values

                        #######################################
                        #####            METE             #####
                        #######################################
                        initial_lambdas = make_initial_guess(X)
                        METE_lambdas = METE(
                            initial_lambdas[:1],
                            {
                                'N/S': float(X['N_t'] / X['S_t'])
                            },
                            X,
                            func_vals[:1],
                            optimizer='trust-constr',
                            maxiter=5e5
                        )
                        METE_lambdas = np.append(METE_lambdas, [0])
                        mete_constraint_errors = check_constraints(METE_lambdas, input_census, func_vals)
                        METE_results, METE_rad = evaluate_model(METE_lambdas, X, func_vals, empirical_rad, mete_constraint_errors)

                        #######################################
                        #####           METimE            #####
                        #######################################
                        best_rad = None
                        best_mae = np.inf

                        for w in [1, 0.1, 10]:
                            print(" ")
                            print("----------METimE----------")
                            METimE_lambdas = run_optimization(METE_lambdas, macro_var, func_vals, slack_weight=w, maxiter=5e5)
                            print("Optimized lambdas: {}".format(METimE_lambdas[:1]))
                            print("Slack variables: {}".format(METimE_lambdas[1:]))
                            metime_constraint_errors = check_constraints(METimE_lambdas, input_census, func_vals)
                            METimE_results, METimE_rad = evaluate_model(METimE_lambdas, X, func_vals, empirical_rad, metime_constraint_errors)
                            print(f"AIC: {METimE_results['AIC'].values[0]}, MAE: {METimE_results['MAE'].values[0]}")

                            if METimE_results["MAE"].values[0] < best_mae:
                                best_mae = METimE_results["MAE"].values[0]
                                best_rad = METimE_rad.copy()

                            ##########################################
                            #####           Save results         #####
                            ##########################################
                            results_list.append({
                                'model': model,
                                'census': census,
                                'slack_weight': w,
                                'N/S': macro_var['N/S'],
                                'dN/S': macro_var['dN/S'],
                                'r2_dn': r2_dn,
                                'METE_error_N/S': mete_constraint_errors[0],
                                'METE_error_dN/S': mete_constraint_errors[1],
                                'METimE_error_N/S': metime_constraint_errors[0],
                                'METimE_error_dN/S': metime_constraint_errors[1],
                                'METE_AIC': METE_results['AIC'].values[0],
                                'METE_MAE': METE_results['MAE'].values[0],
                                'METE_RMSE': METE_results['RMSE'].values[0],
                                'METimE_AIC': METimE_results['AIC'].values[0],
                                'METimE_MAE': METimE_results['MAE'].values[0],
                                'METimE_RMSE': METimE_results['RMSE'].values[0]
                            })

                        per_census_empirical[census].append(empirical_rad)
                        per_census_METE[census].append(METE_rad)
                        per_census_METimE[census].append(best_rad)

                # === AFTER all censuses in the cluster: combine RADs ===
                for census in per_census_empirical.keys():
                    emp_comm, mete_comm, metime_comm = combine_rads_per_census(
                            per_census_empirical[census],
                            per_census_METE[census],
                            per_census_METimE[census]
                    )

                    # evaluate community-level fit
                    comm_METE = evaluate_model_2(emp_comm, mete_comm)
                    comm_METimE = evaluate_model_2(emp_comm, metime_comm)

                    results_list.append({
                        "model": model,
                        "iter": iter,
                        "cluster": cluster,
                        "census": census,
                        "METE_MAE_comm": comm_METE["MAE"].values[0],
                        "METimE_MAE_comm": comm_METimE["MAE"].values[0],
                        "METE_RMSE_comm": comm_METE["RMSE"].values[0],
                        "METimE_RMSE_comm": comm_METimE["RMSE"].values[0],
                    })

                    plot_RADs(emp_comm[0], mete_comm[0], metime_comm[0], save_name=f"model_{ext}_census={census}_clustered", use_log=False)

                results_df = pd.DataFrame(results_list)
                results_df.to_csv(f'results_LV_df{ext}_clustered.csv', index=False)
