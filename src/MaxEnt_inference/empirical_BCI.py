import csv
import os
import gc
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
from scipy.optimize import root_scalar, minimize
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from src.parametrize_transition_functions.SINDy_like_regression import do_polynomial_regression as sindy
from src.MaxEnt_inference.METimE import run_optimization as METE
from src.parametrize_transition_functions.SINDy_like_regression import get_functions, get_function_values

import warnings
warnings.filterwarnings("ignore")


def get_empirical_RAD(file, census):
    # Load relevant data
    df = pd.read_csv(file)

    if 'census' not in df.columns:
        df = df.rename(columns={'t': 'census', 'Species_ID': 'species'})

    df = df[df['census'] == census]
    df = df[['species', 'n']].drop_duplicates()

    # Create rank abundance distribution
    df = df.sort_values(by='n', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    rad = df[['rank', 'n']].rename(columns={'n': 'abundance'})

    return rad

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
        Z = 1e-12

    if np.isinf(Z):
        Z = 1e10

    return Z

def ecosystem_structure_function(lambdas, func_vals, Z):
    lambdas_arr = np.asarray(lambdas).reshape(-1, 1, 1)
    exponent_matrix = np.sum(-lambdas_arr * func_vals, axis=0)

    R = np.exp(exponent_matrix) / Z

    # check if all entries are zero (or numerically close to zero)
    if np.allclose(R, 0):
        # assign uniform probabilities
        R = np.full_like(R, 1.0 / R.size)
        R = R / R.sum()

    if np.isnan(R).any():
        raise ValueError("NaN values found in R — check Z and exponent_matrix ranges.")

    return R

def entropy(vars, func_vals, scales=[1,1,1,1]):
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
    return ((expected_value - target_value) / target_value)

def penalized_entropy(vars, func_vals, macro_var, scales, slack_weight = 1.0):
    if len(macro_var) > 3:
        return (entropy(vars, func_vals, scales) + slack_weight * (
                (get_relative_errors(vars, func_vals, 2, macro_var['dN/S'], scales)) ** 2 +
                (get_relative_errors(vars, func_vals, 3, macro_var['dE/S'], scales)) ** 2))
    else:
        return (entropy(vars, func_vals, scales) + slack_weight * (
                (get_relative_errors(vars, func_vals, 2, macro_var['dN/S'], scales)) ** 2))


def penalized_entropy_grad(vars, func_vals, macro_var, scales, slack_weight = 1.0):
    """
    Gradient of penalized_entropy with respect to the 4 lambda variables.
    """
    # Scale back lambdas
    vars = vars * scales
    lambdas = vars[:func_vals.shape[0]]

    # Partition function and R
    Z = partition_function(lambdas, func_vals)
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    # safe log(R)
    log_R = np.where(R > 0, np.log(R), 0.0)

    # Expected values <f_i>
    expected_values = np.sum(R * func_vals, axis=(1, 2))

    # Precompute multiplier (1 + log_R)
    one_plus_logR = 1.0 + log_R

    # Gradient of entropy part
    grad_entropy = np.zeros(len(lambdas), dtype=float)
    for j in range(len(lambdas)):
        fj = func_vals[j]
        grad_entropy[j] = np.sum(R * (expected_values[j] - fj) * one_plus_logR)

    # Gradient of penalty part
    grad_penalty = np.zeros(len(lambdas), dtype=float)

    if slack_weight != 0.0:

        if len(macro_var) == 3:
            constrained_indices = [2]
        else:
            constrained_indices = [2, 3]

        for c_idx in constrained_indices:
            target_key = 'dN/S' if c_idx == 2 else 'dE/S'
            target = macro_var[target_key]

            expected_c = expected_values[c_idx]

            if target == 0:
                # error is just the expected value
                err_c = expected_c
                d_errc_d_expected = 1.0

            else:
                # relative error (not squared)
                err_c = (expected_c - target) / target
                d_errc_d_expected = 1.0 / target

            # derivative of expected_c w.r.t lambda_j
            f_c = func_vals[c_idx]
            for j in range(len(lambdas)):
                fj = func_vals[j]
                d_expectedc_d_lam_j = np.sum(R * f_c * (expected_values[j] - fj))

                # chain rule
                d_errc_d_lam_j = d_errc_d_expected * d_expectedc_d_lam_j

                # penalty = slack_weight * err_c^2
                grad_penalty[j] += slack_weight * 2.0 * err_c * d_errc_d_lam_j

    grad = grad_entropy + grad_penalty

    # Rescale back
    grad = grad / scales

    return grad

def beta_function(beta, S, N):
    """
    Beta function used to generate the initial guess for Lagrange multipliers.
    """
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta * (N + 1))) * np.log(1.0 / beta) - S / N

def make_initial_guess(X):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR'diag function meteESF.mete.lambda

    :param state_variables: state variables S, S and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    S, N, E = int(X['S_t']), int(X['N_t']), float(X['E_t'])
    interval = [1.0 / N, S / N]

    try:
        beta = root_scalar(beta_function, x0=0.001, args=(S, N), method='brentq', bracket=interval)
        l2 = S / (E - N)
        l1 = beta.root - l2

        if l1 < 0 or l2 < 0:  # Assumption based on "Derivations of the Core Functions of METE": l1 and l2 are strictly positive
            l1 = 0.1  # this assumption comes from somewhere else but not sure where
            l2 = 0.01

    except:
        l1 = 0.1
        l2 = 0.01

    return [l1, l2, 0, 0]

def single_constraint(vars, func_vals, func_index, target_value, scales):
    """
    Vectorized single constraint calculation
    Weight determines the relative weight of the constraint
    """
    # Scale back lambdas
    vars = vars * scales
    lambdas = vars[:func_vals.shape[0]]

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
    f3_vals = func_vals[2, :, :]
    min_f3, max_f3 = f3_vals.min(), f3_vals.max()
    bounds_dn = compute_lambda_bounds(min_f3, max_f3, 100)

    if len(macro_var) > 3:
        f4_vals = func_vals[3, :, :]
        min_f4, max_f4 = f4_vals.min(), f4_vals.max()
        bounds_de = compute_lambda_bounds(min_f4, max_f4, 100)

        # Define scale factors so that parameters are roughly of the same order of magnitude
        values = np.asarray([vars[0], vars[1], bounds_dn[1], bounds_de[1]], dtype=float)

        scales = np.where(values != 0,
                        10.0 ** np.floor(np.log10(np.abs(values))),
                        1.0)
        vars = vars / scales

        bounds = [
            (-18, 18) / scales[0],
            (-18, 18) / scales[1],
            bounds_dn / scales[2],
            bounds_de / scales[3]
        ]

        constraint_order = ['N/S', 'E/S', 'dN/S', 'dE/S']

    else:
        values = np.asarray([vars[0], vars[1], bounds_dn[1]], dtype=float)

        scales = np.where(values != 0,
                          10.0 ** np.floor(np.log10(np.abs(values))),
                          1.0)
        vars = vars / scales

        bounds = [
            (-18, 18) / scales[0],
            (-18, 18) / scales[1],
            bounds_dn / scales[2]
        ]

        # Collect all constraints
        constraint_order = ['N/S', 'E/S', 'dN/S', 'dE/S']

    constraints = [{
        'type': 'eq',
        'fun': lambda vars, F_k=macro_var[name], idx=i:
        single_constraint(vars, func_vals, idx, F_k, scales)
    } for i, name in enumerate(constraint_order[:2])]                                                                   # Only N/S and E/S are strict constraints


    result = minimize(penalized_entropy,
                      vars,
                      jac=penalized_entropy_grad,
                      args=(func_vals, macro_var, scales, slack_weight),
                      constraints=constraints,
                      bounds=bounds,
                      method="trust-constr",
                      options={'maxiter':maxiter,
                               'initial_tr_radius': 0.05,
                               'xtol': 1e-10,
                               'gtol': 1e-12,
                               'disp': True,
                               'verbose': 1})

    optimized_lambdas = result.x * scales

    return optimized_lambdas

def evaluate_model(lambdas, X, func_vals, empirical_rad, constraint_errors):
    Z = partition_function(lambdas[:func_vals.shape[0]], func_vals)
    R = ecosystem_structure_function(lambdas[:func_vals.shape[0]], func_vals, Z)

    # Compute SAD
    sad = np.sum(R, axis=1)

    while len(sad) < int(X['N_t']):
        sad = np.concatenate([sad, np.zeros(int(X['N_t']) - sad.size)])

    print("Sum of sad: {}".format(np.sum(sad)))

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
    S, N, E = (int(input['S_t'].drop_duplicates().iloc[0]),
               int(input['N_t'].drop_duplicates().iloc[0]),
               input['E_t'].drop_duplicates().iloc[0])

    X = {
        'S_t': S,
        'N_t': N,
        'E_t': E
    }

    macro_var = {
        'N/S': X['N_t'] / X['S_t'],
        'E/S': X['E_t'] / X['S_t'],
        'dN/S': input['dN/S'].unique()[0],
        'dE/S': input['dE/S'].unique()[0]
    }

    Z = partition_function(lambdas[:func_vals.shape[0]], func_vals)
    R = ecosystem_structure_function(lambdas[:func_vals.shape[0]], func_vals, Z)

    absolute_errors = []
    percentage_errors = []

    constraint_order = ['N/S', 'E/S', 'dN/S', 'dE/S']
    for i, name in enumerate(constraint_order):
        val = macro_var[name]
        # Expected value of f_k under R
        expected_value = np.sum(R * func_vals[i])

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

def plot_RADs(empirical_rad, METE_rad, METimE_rad, save_name, obs_label="Simulated", weight="", use_log=False):
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
    plt.plot(ranks, empirical_rad, 'o-', color=greyish, markersize=6, linewidth=2, label=obs_label)
    plt.plot(ranks, METE_rad, 's--', color=blueish, markersize=6, linewidth=2, label='METE')
    plt.plot(ranks, METimE_rad, '^--', color=redish, markersize=6, linewidth=2, label='METimE')

    plt.xlabel("Rank", fontsize=16)

    #plt.title(f"Slack weight: {weight}")

    # Handle log scale
    if use_log:
        plt.yscale("log") # base is 10
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        ylabel = "Abundance (log scale)"
    else:
        ylabel = "Abundance"

    plt.ylabel(ylabel, fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)
    #plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = f'{save_name}.png'
    os.makedirs("results", exist_ok=True)

    if save_name is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.savefig('example sad.png', transparent=True)
        plt.show()

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

def add_row(data):
    filename = "results_per_slack_weight.csv"
    file_exists = os.path.isfile(filename)
    columns = ["census", "quad", "slack_weight", "AIC", "RMSE", "MAE", "entropy"]

    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(data)

def add_row_results_list(data, plot):
    filename = f"C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/eBCI/results.csv"
    file_exists = os.path.isfile(filename)

    columns = [
    'quad',
    'census',
    'slack_weight',
    'N/S',
    'E/S',
    'dN/S',
    'dE/S',
    'r2_dn',
    'r2_de',
    'METE_error_N/S',
    'METE_error_E/S',
    'METE_error_dN/S',
    'METE_error_dE/S',
    'METimE_error_N/S',
    'METimE_error_E/S',
    'METimE_error_dN/S',
    'METimE_error_dE/S',
    'METE_AIC',
    'METE_MAE',
    'METE_RMSE',
    'METimE_AIC',
    'METimE_MAE',
    'METimE_RMSE'
    ]

    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(data)

if __name__ == "__main__":
    # Use ext='' for full BCI, or ext='_quadrat_i' for quadrat i data
    for i in [1, 2, 3, 4, 5]:
        ext = f'_quadrat_{i}'
        #ext = ''

        # Load data
        input = pd.read_csv(f'../../data/BCI_regression_library{ext}.csv')
        #plot_trajectories(input)

        # Compute polynomial coefficients
        best_r2_dn, best_alphas, best_betas = -np.inf, [], []
        for lv_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            alphas, r2_dn, betas, r2_de, scaler = sindy(input, lv_ratio=lv_ratio, outlier_removal=True)
            if r2_dn > best_r2_dn:
                best_r2_dn = r2_dn
                best_alphas = alphas
                best_betas = betas

        alphas = best_alphas['Coefficient'].values
        betas = best_betas['Coefficient'].values
        functions = get_functions()

        for census in input['census'].unique():
            print(f"\n Census: {census} \n")
            input_census = input[input['census'] == census]

            X = input_census[[
                'S_t', 'N_t', 'E_t',
            ]].drop_duplicates().iloc[0]

            macro_var = {
                'N/S': float(X['N_t'] / X['S_t']),
                'E/S': float(X['E_t'] / X['S_t']),
                'dN/S': input_census['dN/S'].unique()[0],
                'dE/S': input_census['dE/S'].unique()[0]
            }

            # Get empirical rank abundance distribution
            empirical_rad = get_empirical_RAD(f'../../data/BCI_regression_library{ext}.csv', census)['abundance']

            # Precompute functions(n, e)
            #max_n = int(min(X['N_t'], 1.5 * max(input_census['n'])))
            max_n = int(X['N_t']-X['S_t'])

            if max_n > 1000:
                max_n = int(max(input_census['n']))
            min_e = max(1, -1.5 * input_census['e'].quantile(0.15))
            max_e = min(X['E_t'], 1.5 * input_census['e'].quantile(0.85))

            func_vals, _ = get_function_values(functions, X, alphas, betas, scaler,
                                               [max_n, min_e, max_e],
                                               show_landscape=False,
                                               training_points=input[['n', 'e']].values)

            #######################################
            #####            METE             #####
            #######################################
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
                maxiter=1e5
            )
            METE_lambdas = np.append(METE_lambdas, [0, 0])
            mete_constraint_errors = check_constraints(METE_lambdas, input_census, func_vals)
            METE_results, METE_rad = evaluate_model(METE_lambdas, X, func_vals, empirical_rad, mete_constraint_errors)
            print(
                f"AIC: {METE_results['AIC'].values[0]}, RMSE: {METE_results['RMSE'].values[0]}, MAE: {METE_results['MAE'].values[0]}")
            #METE_lambdas = np.append(METE_lambdas, [0, 0])

            #######################################
            #####           METimE            #####
            #######################################
            prev_best_MAE = np.inf
            for w in [100, 10, 1, 0.1, 0]:
                print(" ")
                print("----------METimE----------")
                METimE_lambdas = run_optimization(METE_lambdas, macro_var, func_vals, slack_weight=w, maxiter=1e5)
                print("Optimized lambdas: {}".format(METimE_lambdas[:4]))
                metime_constraint_errors = check_constraints(METimE_lambdas, input_census, func_vals)
                METimE_results, METimE_rad = evaluate_model(METimE_lambdas, X, func_vals, empirical_rad, metime_constraint_errors)
                print(f"AIC: {METimE_results['AIC'].values[0]}, RMSE: {METimE_results['RMSE'].values[0]}, MAE: {METimE_results['MAE'].values[0]}")

                ##########################################
                #####           Save results         #####
                ##########################################
                add_row_results_list({
                        'quad': ext,
                        'census': census,
                        'slack_weight': w,
                        'N/S': macro_var['N/S'],
                        'E/S': macro_var['E/S'],
                        'dN/S': macro_var['dN/S'],
                        'dE/S': macro_var['dE/S'],
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
                    #     "census": census,
                    #     "quad": ext,
                    #     "slack_weight": w,
                    #     "AIC": METimE_results['AIC'].values[0],
                    #     "RMSE": METimE_results['RMSE'].values[0],
                    #     "MAE": METimE_results['MAE'].values[0],
                    #     "entropy": -entropy(METimE_lambdas[:4], func_vals)
                    # })

                if METimE_results['MAE'].values[0] < prev_best_MAE:
                    plot_RADs(empirical_rad, METE_rad, METimE_rad, f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/eBCI/quad_{ext}_census_{census}.png', 'Empirical', weight={w}, use_log=True)
                    prev_best_MAE = METimE_results['MAE'].values[0]
            gc.collect()
        # results_df = pd.DataFrame(results_list)
        # results_df.to_csv(f'20_5_empirical_BCI_df{ext}.csv', index=False)