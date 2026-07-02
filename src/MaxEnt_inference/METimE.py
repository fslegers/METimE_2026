import os
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize
from scipy.stats import rv_discrete
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from src.parametrize_transition_functions.SINDy_like_regression import do_polynomial_regression as sindy

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

def partition_function(lambdas, func_vals):
    lambdas = np.asarray(lambdas).reshape(-1, 1, 1)
    exponent_matrix = np.sum(-lambdas * func_vals, axis=0)
    Z = np.exp(exponent_matrix).sum()

    if np.isclose(Z, 0.0, atol=1e-12):
        print("Invalid values detected in Z")

    if np.isinf(Z):
        print("Invalid values detected in Z")
        Z = 1e10

    return Z

def ecosystem_structure_function(lambdas, func_vals, Z):
    lambdas_arr = np.asarray(lambdas).reshape(-1, 1, 1)
    exponent_matrix = np.sum(-lambdas_arr * func_vals, axis=0)
    R = np.exp(exponent_matrix) / Z

    if np.isnan(R).any():
        raise ValueError("NaN values found in R — check Z and exponent_matrix ranges.")

    return R

def entropy(lambdas, func_vals, scales=[1,1,1,1]):
    """
    Computes Shannon entropy: -sum R(n,e) * log(R(n,e))
    """
    # Scale back lambdas
    lambdas = lambdas * scales

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

def entropy_grad(lambdas, func_vals, scales=[1,1,1,1]):
    # Scale back lambdas
    lambdas = lambdas * scales

    # Partition function
    Z = partition_function(lambdas, func_vals)

    # Ecosystem structure function
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    # log(R) (safe)
    log_R = np.where(R > 0, np.log(R), 0.0)

    # Expected values <f_i> under R
    expected_values = np.sum(R * func_vals, axis=(1, 2))  # shape (num_funcs,)

    grad = np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        fi = func_vals[i]  # same shape as R
        grad[i] = np.sum(R * (expected_values[i] - fi) * log_R)

    # Rescale back to the original parameterization
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

def single_constraint(lambdas, func_vals, func_index, target_value, weight = 1, scales=[1,1,1,1]):
    """
    Vectorized single constraint calculation
    Weight determines the relative weight of the constraint
    """
    # Scale back lambdas
    lambdas = lambdas * scales

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
    return np.abs(expected_value - target_value) / (target_value) * weight

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

def run_optimization(lambdas, macro_var, X, func_vals, maxiter=1000, optimizer='trust-constr', gtol=1e-12):
    lambdas = np.asarray(lambdas, dtype=float)

    if len(lambdas) == 4:
        # Compute bounds (to prevent overflow in exp)
        f3_vals = func_vals[2, :, :]  # shape (N, len(e_vals))
        f4_vals = func_vals[3, :, :]
        min_f3, max_f3 = f3_vals.min(), f3_vals.max()
        min_f4, max_f4 = f4_vals.min(), f4_vals.max()
        bounds_dn = compute_lambda_bounds(min_f3, max_f3, 100)
        bounds_de = compute_lambda_bounds(min_f4, max_f4, 100)

        # Define scale factors so that parameters are roughly of the same order of magnitude
        values = np.asarray([lambdas[0], lambdas[1], bounds_dn[1], bounds_de[1]], dtype=float)

    elif len(lambdas) == 2:
        values = np.asarray([lambdas[0], lambdas[1]], dtype=float)

    else:
        values = np.asarray([lambdas[0]], dtype=float)

    scales = np.where(values != 0,
                    10.0 ** np.floor(np.log10(np.abs(values))),
                    1.0)
    lambdas = lambdas / scales

    if len(lambdas) == 4:
        bounds = [(-18, 18) / scales[0],                                                                              # TODO: changed this to -18 for METE simulated BCI
                  (-18, 18) / scales[1],
                  bounds_dn / scales[2],
                  bounds_de / scales[3]]
        weights = [1, 1, 0.01, 0.01]
    elif len(lambdas) == 2:
        bounds = [(-18, 18) / scales[0],                                                                              # TODO: changed this to -18 for METE simulated BCI
                  (-18, 18) / scales[1]]
        weights = [1, 1]
    else:
        bounds = [(-18, 18) / scales[0]]
        weights = [1]

    # Collect all constraints
    constraint_order = ['N/S', 'E/S', 'dN/S', 'dE/S'][:len(lambdas)]

    constraints = [{
        'type': 'eq',
        'fun': lambda lambdas, F_k=macro_var[name], idx=i:
        single_constraint(lambdas, func_vals, idx, F_k, weights[i], scales)
    } for i, name in enumerate(constraint_order)]

    if optimizer == 'trust-constr':
        result = minimize(entropy,
                          lambdas,
                          args=(func_vals, scales),
                          constraints=constraints,
                          bounds=bounds[:len(lambdas)],
                          method="trust-constr",
                          options={'maxiter': maxiter,
                                   'initial_tr_radius': 0.1,
                                   'xtol': 1e-10,
                                   'gtol': gtol,
                                   'disp': True,
                                   'verbose': 1
                                   })

    else:
        result = minimize(entropy,
                          lambdas,
                          args=(func_vals, scales),
                          constraints=constraints,
                          method="SLSQP",
                          options={'disp': True,
                          'verbose': 3})

    optimized_lambdas = result.x * scales

    return optimized_lambdas

def f_n(n, e, X, alphas, betas):
    return n

def f_ne(n, e, X, alphas, betas):
    return n * e

def f_dn(n, e, X, alphas, betas):
    features = [e, (X['S_t']), n, 1, (e ** (-1)), (e ** (-3/4)), (e ** (-1/2)), (e ** (1/4)), (e ** (1/2)),
         (e ** (3/4)), (e), (e ** (3/2)), (n ** (1/2)), (n), (n ** (3/4)), (n ** (3/2)), (np.log(n)),
         (np.log(e)), (X['N_t']), (1/X['N_t']), (X['E_t']), (1/X['E_t']), (e ** (-1))**2,
         (e ** (-1))*(e ** (-3/4)), (e ** (-1))*(e ** (-1/2)), (e ** (-1))*(e ** (1/4)),
         (e ** (-1))*(e ** (1/2)), (e ** (-1))*(e ** (3/4)), (e ** (-1))*(e), (e ** (-1))*(e ** (3/2)),
         (e ** (-1))*(n ** (1/2)), (e ** (-1))*(n), (e ** (-1))*(n ** (3/4)), (e ** (-1))*(n ** (3/2)),
         (e ** (-1))*(np.log(n)), (e ** (-1))*(np.log(e)), (e ** (-1))*(X['N_t']), (e ** (-1))*(1/X['N_t']),
         (e ** (-1))*(X['E_t']), (e ** (-1))*(1/X['E_t']), (e ** (-3/4))**2, (e ** (-3/4))*(e ** (-1/2)),
         (e ** (-3/4))*(e ** (1/4)), (e ** (-3/4))*(e ** (1/2)), (e ** (-3/4))*(e ** (3/4)), (e ** (-3/4))*(e),
         (e ** (-3/4))*(e ** (3/2)), (e ** (-3/4))*(n ** (1/2)), (e ** (-3/4))*(n), (e ** (-3/4))*(n ** (3/4)),
         (e ** (-3/4))*(n ** (3/2)), (e ** (-3/4))*(np.log(n)), (e ** (-3/4))*(np.log(e)), (e ** (-3/4))*(X['N_t']),
         (e ** (-3/4))*(1/X['N_t']), (e ** (-3/4))*(X['E_t']), (e ** (-3/4))*(1/X['E_t']),
         (e ** (-1/2))**2, (e ** (-1/2))*(e ** (1/4)), (e ** (-1/2))*(e ** (1/2)), (e ** (-1/2))*(e ** (3/4)),
         (e ** (-1/2))*(e), (e ** (-1/2))*(e ** (3/2)), (e ** (-1/2))*(n ** (1/2)), (e ** (-1/2))*(n),
         (e ** (-1/2))*(n ** (3/4)), (e ** (-1/2))*(n ** (3/2)), (e ** (-1/2))*(np.log(n)),
         (e ** (-1/2))*(np.log(e)), (e ** (-1/2))*(X['N_t']), (e ** (-1/2))*(1/X['N_t']), (e ** (-1/2))*(X['E_t']),
         (e ** (-1/2))*(1/X['E_t']), (e ** (1/4))**2, (e ** (1/4))*(e ** (1/2)),
         (e ** (1/4))*(e ** (3/4)), (e ** (1/4))*(e), (e ** (1/4))*(e ** (3/2)), (e ** (1/4))*(n ** (1/2)),
         (e ** (1/4))*(n), (e ** (1/4))*(n ** (3/4)), (e ** (1/4))*(n ** (3/2)), (e ** (1/4))*(np.log(n)),
         (e ** (1/4))*(np.log(e)), (e ** (1/4))*(X['N_t']), (e ** (1/4))*(1/X['N_t']), (e ** (1/4))*(X['E_t']),
         (e ** (1/4))*(1/X['E_t']), (e ** (1/2))**2, (e ** (1/2))*(e ** (3/4)), (e ** (1/2))*(e),
         (e ** (1/2))*(e ** (3/2)), (e ** (1/2))*(n ** (1/2)), (e ** (1/2))*(n), (e ** (1/2))*(n ** (3/4)),
         (e ** (1/2))*(n ** (3/2)), (e ** (1/2))*(np.log(n)), (e ** (1/2))*(np.log(e)), (e ** (1/2))*(X['N_t']),
         (e ** (1/2))*(1/X['N_t']), (e ** (1/2))*(X['E_t']), (e ** (1/2))*(1/X['E_t']), (e ** (3/4))**2,
         (e ** (3/4))*(e), (e ** (3/4))*(e ** (3/2)), (e ** (3/4))*(n ** (1/2)), (e ** (3/4))*(n),
         (e ** (3/4))*(n ** (3/4)), (e ** (3/4))*(n ** (3/2)), (e ** (3/4))*(np.log(n)),
         (e ** (3/4))*(np.log(e)), (e ** (3/4))*(X['N_t']), (e ** (3/4))*(1/X['N_t']), (e ** (3/4))*(X['E_t']),
         (e ** (3/4))*(1/X['E_t']), (e)**2, (e)*(e ** (3/2)), (e)*(n ** (1/2)), (e)*(n), (e)*(n ** (3/4)),
         (e)*(n ** (3/2)), (e)*(np.log(n)), (e)*(np.log(e)), (e)*(X['N_t']), (e)*(1/X['N_t']), (e)*(X['E_t']),
         (e)*(1/X['E_t']), (e ** (3/2))**2, (e ** (3/2))*(n ** (1/2)), (e ** (3/2))*(n),
         (e ** (3/2))*(n ** (3/4)), (e ** (3/2))*(n ** (3/2)), (e ** (3/2))*(np.log(n)),
         (e ** (3/2))*(np.log(e)), (e ** (3/2))*(X['N_t']), (e ** (3/2))*(1/X['N_t']), (e ** (3/2))*(X['E_t']),
         (e ** (3/2))*(1/X['E_t']), (n ** (1/2))**2, (n ** (1/2))*(n), (n ** (1/2))*(n ** (3/4)),
         (n ** (1/2))*(n ** (3/2)), (n ** (1/2))*(np.log(n)), (n ** (1/2))*(np.log(e)), (n ** (1/2))*(X['N_t']),
         (n ** (1/2))*(1/X['N_t']), (n ** (1/2))*(X['E_t']), (n ** (1/2))*(1/X['E_t']), (n)**2,
         (n)*(n ** (3/4)), (n)*(n ** (3/2)), (n)*(np.log(n)), (n)*(np.log(e)), (n)*(X['N_t']), (n)*(1/X['N_t']),
         (n)*(X['E_t']), (n)*(1/X['E_t']), (n ** (3/4))**2, (n ** (3/4))*(n ** (3/2)),
         (n ** (3/4))*(np.log(n)), (n ** (3/4))*(np.log(e)), (n ** (3/4))*(X['N_t']), (n ** (3/4))*(1/X['N_t']),
         (n ** (3/4))*(X['E_t']), (n ** (3/4))*(1/X['E_t']), (n ** (3/2))**2, (n ** (3/2))*(np.log(n)),
         (n ** (3/2))*(np.log(e)), (n ** (3/2))*(X['N_t']), (n ** (3/2))*(1/X['N_t']), (n ** (3/2))*(X['E_t']),
         (n ** (3/2))*(1/X['E_t']), (np.log(n))**2, (np.log(n))*(np.log(e)), (np.log(n))*(X['N_t']),
         (np.log(n))*(1/X['N_t']), (np.log(n))*(X['E_t']), (np.log(n))*(1/X['E_t']), (np.log(e))**2,
         (np.log(e))*(X['N_t']), (np.log(e))*(1/X['N_t']), (np.log(e))*(X['E_t']), (np.log(e))*(1/X['E_t']),
         (X['N_t'])**2, (X['N_t'])*(1/X['N_t']), (X['N_t'])*(X['E_t']), (X['N_t'])*(1/X['E_t']), (1/X['N_t'])**2,
         (1/X['N_t'])*(X['E_t']), (1/X['N_t'])*(1/X['E_t']), (X['E_t'])**2, (X['E_t'])*(1/X['E_t']),
         (1/X['E_t'])**2]
    result = sum(alpha * f for alpha, f in zip(alphas, features))
    return np.maximum(result, -n)

def f_de(n, e, X, alphas, betas):
    X = dict(X)
    features = [e, (X['S_t']), n, 1, (e ** (-1)), (e ** (-3/4)), (e ** (-1/2)), (e ** (1/4)), (e ** (1/2)),
         (e ** (3/4)), (e), (e ** (3/2)), (n ** (1/2)), (n), (n ** (3/4)), (n ** (3/2)), (np.log(n)),
         (np.log(e)), (X['N_t']), (1/X['N_t']), (X['E_t']), (1/X['E_t']), (e ** (-1))**2,
         (e ** (-1))*(e ** (-3/4)), (e ** (-1))*(e ** (-1/2)), (e ** (-1))*(e ** (1/4)),
         (e ** (-1))*(e ** (1/2)), (e ** (-1))*(e ** (3/4)), (e ** (-1))*(e), (e ** (-1))*(e ** (3/2)),
         (e ** (-1))*(n ** (1/2)), (e ** (-1))*(n), (e ** (-1))*(n ** (3/4)), (e ** (-1))*(n ** (3/2)),
         (e ** (-1))*(np.log(n)), (e ** (-1))*(np.log(e)), (e ** (-1))*(X['N_t']), (e ** (-1))*(1/X['N_t']),
         (e ** (-1))*(X['E_t']), (e ** (-1))*(1/X['E_t']), (e ** (-3/4))**2, (e ** (-3/4))*(e ** (-1/2)),
         (e ** (-3/4))*(e ** (1/4)), (e ** (-3/4))*(e ** (1/2)), (e ** (-3/4))*(e ** (3/4)), (e ** (-3/4))*(e),
         (e ** (-3/4))*(e ** (3/2)), (e ** (-3/4))*(n ** (1/2)), (e ** (-3/4))*(n), (e ** (-3/4))*(n ** (3/4)),
         (e ** (-3/4))*(n ** (3/2)), (e ** (-3/4))*(np.log(n)), (e ** (-3/4))*(np.log(e)), (e ** (-3/4))*(X['N_t']),
         (e ** (-3/4))*(1/X['N_t']), (e ** (-3/4))*(X['E_t']), (e ** (-3/4))*(1/X['E_t']),
         (e ** (-1/2))**2, (e ** (-1/2))*(e ** (1/4)), (e ** (-1/2))*(e ** (1/2)), (e ** (-1/2))*(e ** (3/4)),
         (e ** (-1/2))*(e), (e ** (-1/2))*(e ** (3/2)), (e ** (-1/2))*(n ** (1/2)), (e ** (-1/2))*(n),
         (e ** (-1/2))*(n ** (3/4)), (e ** (-1/2))*(n ** (3/2)), (e ** (-1/2))*(np.log(n)),
         (e ** (-1/2))*(np.log(e)), (e ** (-1/2))*(X['N_t']), (e ** (-1/2))*(1/X['N_t']), (e ** (-1/2))*(X['E_t']),
         (e ** (-1/2))*(1/X['E_t']), (e ** (1/4))**2, (e ** (1/4))*(e ** (1/2)),
         (e ** (1/4))*(e ** (3/4)), (e ** (1/4))*(e), (e ** (1/4))*(e ** (3/2)), (e ** (1/4))*(n ** (1/2)),
         (e ** (1/4))*(n), (e ** (1/4))*(n ** (3/4)), (e ** (1/4))*(n ** (3/2)), (e ** (1/4))*(np.log(n)),
         (e ** (1/4))*(np.log(e)), (e ** (1/4))*(X['N_t']), (e ** (1/4))*(1/X['N_t']), (e ** (1/4))*(X['E_t']),
         (e ** (1/4))*(1/X['E_t']), (e ** (1/2))**2, (e ** (1/2))*(e ** (3/4)), (e ** (1/2))*(e),
         (e ** (1/2))*(e ** (3/2)), (e ** (1/2))*(n ** (1/2)), (e ** (1/2))*(n), (e ** (1/2))*(n ** (3/4)),
         (e ** (1/2))*(n ** (3/2)), (e ** (1/2))*(np.log(n)), (e ** (1/2))*(np.log(e)), (e ** (1/2))*(X['N_t']),
         (e ** (1/2))*(1/X['N_t']), (e ** (1/2))*(X['E_t']), (e ** (1/2))*(1/X['E_t']), (e ** (3/4))**2,
         (e ** (3/4))*(e), (e ** (3/4))*(e ** (3/2)), (e ** (3/4))*(n ** (1/2)), (e ** (3/4))*(n),
         (e ** (3/4))*(n ** (3/4)), (e ** (3/4))*(n ** (3/2)), (e ** (3/4))*(np.log(n)),
         (e ** (3/4))*(np.log(e)), (e ** (3/4))*(X['N_t']), (e ** (3/4))*(1/X['N_t']), (e ** (3/4))*(X['E_t']),
         (e ** (3/4))*(1/X['E_t']), (e)**2, (e)*(e ** (3/2)), (e)*(n ** (1/2)), (e)*(n), (e)*(n ** (3/4)),
         (e)*(n ** (3/2)), (e)*(np.log(n)), (e)*(np.log(e)), (e)*(X['N_t']), (e)*(1/X['N_t']), (e)*(X['E_t']),
         (e)*(1/X['E_t']), (e ** (3/2))**2, (e ** (3/2))*(n ** (1/2)), (e ** (3/2))*(n),
         (e ** (3/2))*(n ** (3/4)), (e ** (3/2))*(n ** (3/2)), (e ** (3/2))*(np.log(n)),
         (e ** (3/2))*(np.log(e)), (e ** (3/2))*(X['N_t']), (e ** (3/2))*(1/X['N_t']), (e ** (3/2))*(X['E_t']),
         (e ** (3/2))*(1/X['E_t']), (n ** (1/2))**2, (n ** (1/2))*(n), (n ** (1/2))*(n ** (3/4)),
         (n ** (1/2))*(n ** (3/2)), (n ** (1/2))*(np.log(n)), (n ** (1/2))*(np.log(e)), (n ** (1/2))*(X['N_t']),
         (n ** (1/2))*(1/X['N_t']), (n ** (1/2))*(X['E_t']), (n ** (1/2))*(1/X['E_t']), (n)**2,
         (n)*(n ** (3/4)), (n)*(n ** (3/2)), (n)*(np.log(n)), (n)*(np.log(e)), (n)*(X['N_t']), (n)*(1/X['N_t']),
         (n)*(X['E_t']), (n)*(1/X['E_t']), (n ** (3/4))**2, (n ** (3/4))*(n ** (3/2)),
         (n ** (3/4))*(np.log(n)), (n ** (3/4))*(np.log(e)), (n ** (3/4))*(X['N_t']), (n ** (3/4))*(1/X['N_t']),
         (n ** (3/4))*(X['E_t']), (n ** (3/4))*(1/X['E_t']), (n ** (3/2))**2, (n ** (3/2))*(np.log(n)),
         (n ** (3/2))*(np.log(e)), (n ** (3/2))*(X['N_t']), (n ** (3/2))*(1/X['N_t']), (n ** (3/2))*(X['E_t']),
         (n ** (3/2))*(1/X['E_t']), (np.log(n))**2, (np.log(n))*(np.log(e)), (np.log(n))*(X['N_t']),
         (np.log(n))*(1/X['N_t']), (np.log(n))*(X['E_t']), (np.log(n))*(1/X['E_t']), (np.log(e))**2,
         (np.log(e))*(X['N_t']), (np.log(e))*(1/X['N_t']), (np.log(e))*(X['E_t']), (np.log(e))*(1/X['E_t']),
         (X['N_t'])**2, (X['N_t'])*(1/X['N_t']), (X['N_t'])*(X['E_t']), (X['N_t'])*(1/X['E_t']), (1/X['N_t'])**2,
         (1/X['N_t'])*(X['E_t']), (1/X['N_t'])*(1/X['E_t']), (X['E_t'])**2, (X['E_t'])*(1/X['E_t']),
         (1/X['E_t'])**2]
    result = sum(beta * f for beta, f in zip(betas, features))
    return np.maximum(result, -e)

def get_functions(coef_dn, coef_de):
    return [f_n, f_ne, f_dn, f_de]

def get_function_values(functions, X, alphas, betas, de, show_landscape=False, training_points=None):
    e_vals = np.arange(1, X['E_t'] + de, de, dtype=float)
    n_vals = np.arange(1, int(X['N_t']) + 1, dtype=float)

    # Create grid
    n_grid, e_grid = np.meshgrid(n_vals, e_vals, indexing='ij')  # shape: (N, len_e_vals)

    num_funcs = len(functions)
    results = np.zeros((num_funcs, len(n_vals), len(e_vals)))

    for i, func in enumerate(functions):
        results[i] = func(n_grid, e_grid, X, alphas, betas)

        # Plotting
        if show_landscape:
            tp = None
            if training_points is not None:
                tp = np.array(training_points)
                n_min, n_max = tp[:, 0].min(), tp[:, 0].max()
                e_min, e_max = tp[:, 1].min(), tp[:, 1].max()
                # add 10% padding
                n_pad = 0.1 * (n_max - n_min)
                e_pad = 0.1 * (e_max - e_min)
                n_min, n_max = n_min - n_pad, n_max + n_pad
                e_min, e_max = e_min - e_pad, e_max + e_pad

            # Make a grid of 2 rows: [global view, zoomed view]
            fig, axes = plt.subplots(2, num_funcs,
                                     subplot_kw={"projection": "3d"},
                                     figsize=(6 * num_funcs, 10))

            if num_funcs == 1:  # if only one function
                axes = np.array([[axes[0]], [axes[1]]])

            for i in range(num_funcs):
                # --- Global plot ---
                ax_global = axes[0, i]
                surf = ax_global.plot_surface(
                    n_grid, e_grid, results[i],
                    cmap="viridis", edgecolor="none", alpha=0.9
                )
                ax_global.set_title(f"Function {i + 1} (Global)")
                ax_global.set_xlabel("n")
                ax_global.set_ylabel("e")
                ax_global.set_zlabel("f(n,e)")
                fig.colorbar(surf, ax=ax_global, shrink=0.5, aspect=10)

                if tp is not None:
                    # Evaluate training points at actual height
                    z_tp = functions[i](tp[:, 0], tp[:, 1], X, alphas, betas)
                    ax_global.scatter(tp[:, 0], tp[:, 1], z_tp,
                                      c="red", marker="o", s=40,
                                      label="Training points")
                    ax_global.legend()

                # --- Zoomed plot ---
                ax_zoom = axes[1, i]

                if tp is not None:
                    # Mask grid to zoom region
                    mask = ((n_grid >= n_min) & (n_grid <= n_max) &
                            (e_grid >= e_min) & (e_grid <= e_max))
                    n_zoom = np.where(mask, n_grid, np.nan)
                    e_zoom = np.where(mask, e_grid, np.nan)
                    z_zoom = np.where(mask, results[i], np.nan)

                    surf_zoom = ax_zoom.plot_surface(
                        n_zoom, e_zoom, z_zoom,
                        cmap="viridis", edgecolor="none", alpha=0.9
                    )
                    ax_zoom.set_xlim(n_min, n_max)
                    ax_zoom.set_ylim(e_min, e_max)
                    ax_zoom.set_title(f"Function {i + 1} (Zoom)")
                    ax_zoom.set_xlabel("n")
                    ax_zoom.set_ylabel("e")
                    ax_zoom.set_zlabel("f(n,e)")
                    fig.colorbar(surf_zoom, ax=ax_zoom, shrink=0.5, aspect=10)

                    # Training points at actual height
                    z_tp = functions[i](tp[:, 0], tp[:, 1], X, alphas, betas)
                    ax_zoom.scatter(tp[:, 0], tp[:, 1], z_tp,
                                    c="red", marker="o", s=40,
                                    label="Training points")
                    ax_zoom.legend()

            plt.tight_layout()
            plt.show()

    return results, e_vals

def do_polynomial_regression(df, show_plot=False, show_landscape=False):
    # Select the columns to apply polynomial features
    poly_cols = ['e', 'n', 'S_t', 'N_t', 'E_t']

    # Generate polynomial features
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly_features = poly.fit_transform(df[poly_cols])

    # Create a new DataFrame with polynomial features
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    # Concatenate polynomial features back to the original DataFrame
    df = pd.concat([df.drop(columns=poly_cols), poly_df], axis=1)

    # Drop 'tree_id' and dN/S and dE/S columns
    df = df.drop(columns=['TreeID', 'dN/S', 'dE/S', 'dS'], errors='ignore')

    # Group by (t, species_id) and sum all features
    df_grouped = df.groupby(['census', 'species']).sum().reset_index()

    # Now fit the linear regression model
    dn_obs = df_grouped['dn']
    de_obs = df_grouped['de']
    X = df_grouped.drop(columns=['census', 'species', 'dn', 'de'])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    output = []
    models = []
    stats = {"training": {}, "grid": {}}
    for y in [dn_obs, de_obs]:
        model = LinearRegression()
        model.fit(X_scaled, y)
        models.append(model)
        y_pred = model.predict(X_scaled)

        # De-standardize coefficients
        beta_std = model.coef_
        mu = scaler.mean_
        sigma = scaler.scale_

        beta_orig = beta_std / sigma
        intercept_orig = model.intercept_ - np.sum((beta_std * mu) / sigma)

        # Combine into DataFrame
        coeff_df = pd.DataFrame({
            'Feature': poly_feature_names,
            'Coefficient': beta_orig
        })

        # Add intercept as a separate row (optional but useful)
        coeff_df.loc[len(coeff_df)] = ['Intercept', intercept_orig]

        # Calculate r2
        r2 = r2_score(y, y_pred)

        output.append(coeff_df)
        output.append(r2)

        if show_plot:
            plt.scatter(y, y_pred)
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.title(f"Polynomial Regression: R2 = {r2:.2f}")
            plt.show()

    if show_landscape:
        # Build grid over n,e space
        n_min, n_max = df_grouped['n'].min(), df_grouped['n'].max()
        e_min, e_max = df_grouped['e'].min(), df_grouped['e'].max()
        n_grid, e_grid = np.meshgrid(
            np.linspace(n_min, n_max, 30),
            np.linspace(e_min, e_max, 30)
        )

        # Use median values for other variables (S_t, N_t, E_t)
        S_med = df_grouped['S_t'].median()
        N_med = df_grouped['N_t'].median()
        E_med = df_grouped['E_t'].median()

        # Flatten and create feature matrix
        grid_points = pd.DataFrame({
            'e': e_grid.ravel(),
            'n': n_grid.ravel(),
            'S_t': S_med,
            'N_t': N_med,
            'E_t': E_med
        })

        # Apply polynomial transform
        grid_poly = poly.transform(grid_points)
        grid_poly_df = pd.DataFrame(grid_poly, columns=poly_feature_names)

        # Scale
        grid_scaled = scaler.transform(grid_poly_df)

        # Predict dn and de landscapes (with lower bounds -n and -e)
        dn_pred = models[0].predict(grid_scaled).reshape(n_grid.shape)
        dn_pred = np.maximum(dn_pred, -df_grouped['n'].max())

        de_pred = models[1].predict(grid_scaled).reshape(n_grid.shape)
        de_pred = np.maximum(de_pred, -df_grouped['e'].max())

        # Grid stats
        for label, pred in zip(["dn", "de"], [dn_pred, de_pred]):
            stats["grid"][label] = {
                "mean_pred": float(np.mean(pred)),
                "std_pred": float(np.std(pred)),
                "min_pred": float(np.min(pred)),
                "max_pred": float(np.max(pred))
            }

            stats["training"][label] = {
                "mean_pred": np.mean(y_pred),
                "std_pred": np.std(y_pred),
                "min_pred": np.min(y_pred),
                "max_pred": np.max(y_pred)
            }

        print(stats)

        # Plot landscapes side by side
        fig = plt.figure(figsize=(16, 7))

        for i, (z_pred, label, obs) in enumerate(zip(
                [dn_pred, de_pred],
                ['dn', 'de'],
                [dn_obs, de_obs]
        )):
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            ax.plot_surface(n_grid, e_grid, z_pred, cmap='viridis', alpha=0.7)
            ax.scatter(df_grouped['n'], df_grouped['e'], obs, color='red')
            ax.set_xlabel('n')
            ax.set_ylabel('e')
            ax.set_zlabel(label)
            ax.set_title(f"Landscape for {label}")

        plt.show()

    return output

def evaluate_model(lambdas, X, func_vals, empirical_rad, de, constraint_errors):
    Z = partition_function(lambdas, func_vals)
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    # Compute SAD
    sad = np.sum(R, axis=1)
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

    Z = partition_function(lambdas, func_vals)
    R = ecosystem_structure_function(lambdas, func_vals, Z)

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
    plt.plot(ranks, empirical_rad, 'o-', color=greyish, markersize=6, linewidth=2, label='Empirical')
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

    # 3. Plot S_t
    axes[2].plot(df['census'].unique(), df.groupby('census')['S_t'].mean(), marker='o')
    axes[2].set_title('S_t')

    # 4. Plot N_t
    axes[3].plot(df['census'].unique(), df.groupby('census')['N_t'].mean(), marker='o')
    axes[3].set_title('N_t')

    # 5. Plot E_t
    axes[4].plot(df['census'].unique(), df.groupby('census')['E_t'].mean(), marker='o')
    axes[4].set_title('E_t')
    axes[4].set_xlabel('Census')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Use ext='' for full BCI, or ext='_quadrat_i' for quadrat i data
    for i in [1, 2, 3, 4]:
        ext = f'_quadrat_{i}'

        # Load data
        input = pd.read_csv(f'../../data/BCI_regression_library{ext}.csv')
        plot_trajectories(input)

        # Compute polynomial coefficients
        alphas, r2_dn, betas, r2_de = sindy(input)
        print()
        print("Coefficients dn:")
        print(alphas)
        print("Coefficients de:")
        print(betas)
        print()
        functions = get_functions(alphas, betas)
        r2s = pd.DataFrame({'r2_dn': [r2_dn], 'r2_de': [r2_de]})
        r2s.to_csv(f'empirical_BCI_r2_tf{ext}.csv', index=False)
        alphas = alphas['Coefficient'].values
        betas = betas['Coefficient'].values

        # Create list to store results
        results_list = []

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
            de = 1
            func_vals, _ = get_function_values(functions, X, alphas, betas, de,
                                               show_landscape=True,
                                               training_points=input[['n', 'e']].values)

            # Make initial guess
            initial_lambdas = make_initial_guess(X)
            print(f"Initial guess (theoretical): {initial_lambdas}")

            #######################################
            #####            METE             #####
            #######################################
            print(" ")
            print("----------METE----------")
            METE_lambdas = run_optimization(
                initial_lambdas[:2],
                {
                    'N/S': float(X['N_t'] / X['S_t']),
                    'E/S': float(X['E_t'] / X['S_t'])
                },
                X,
                func_vals[:2],
                de,
                optimizer='trust-constr'
            )
            print("Optimized lambdas (METE): {}".format(METE_lambdas))
            METE_lambdas = np.append(METE_lambdas, [0, 0])
            constraint_errors = check_constraints(METE_lambdas, input_census, func_vals)
            METE_results, METE_rad = evaluate_model(METE_lambdas, X, func_vals, empirical_rad, de, constraint_errors)
            print(f"AIC: {METE_results['AIC'].values[0]}, MAE: {METE_results['MAE'].values[0]}")

            #######################################
            #####           METimE            #####
            #######################################
            print(" ")
            print("----------METimE----------")
            METimE_lambdas = run_optimization(METE_lambdas, macro_var, X, func_vals, de, optimizer='trust-constr')
            print("Optimized lambdas: {}".format(METimE_lambdas))
            constraint_errors = check_constraints(METimE_lambdas, input_census, func_vals)
            METimE_results, METimE_rad = evaluate_model(METimE_lambdas, X, func_vals, empirical_rad, de, constraint_errors)
            print(f"AIC: {METimE_results['AIC'].values[0]}, MAE: {METimE_results['MAE'].values[0]}")

            ##########################################
            #####           Save results         #####
            ##########################################
            results_list.append({
                'quad': ext,
                'census': census,
                'N/S': macro_var['N/S'],
                'E/S': macro_var['E/S'],
                'dN/S': macro_var['dN/S'],
                'dE/S': macro_var['dE/S'],
                'r2_dn': r2_dn,
                'r2_de': r2_de,
                'METE_AIC': METE_results['AIC'].values[0],
                'METE_MAE': METE_results['MAE'].values[0],
                'METE_RMSE': METE_results['RMSE'].values[0],
                'METimE_AIC': METimE_results['AIC'].values[0],
                'METimE_MAE': METimE_results['MAE'].values[0],
                'METimE_RMSE': METimE_results['RMSE'].values[0]
            })

            plot_RADs(empirical_rad, METE_rad, METimE_rad, f'quad_{ext}_census_{census}', use_log=True)

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'empirical_BCI_df{ext}.csv', index=False)