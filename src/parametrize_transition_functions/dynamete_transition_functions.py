import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def get_transition_function_accuracy(df):
    y_obs = np.array('dn')
    y_pred = f_dn(df['n'], df['e'], {'N': df['N_t'], 'E': df['E_t']})
    r2_dn  = r2_score(y_obs, y_pred)

    y_obs = np.array('de')
    y_pred = f_de(df['n'], df['e'], {'N': df['N'], 'E': df['E']})
    r2_de  = r2_score(y_obs, y_pred)

    return r2_dn, r2_de

def f_n(n, e, X):
    return n

def f_ne(n, e, X):
    return n * e

def f_dn(n, e, X):
    b, d, Ec, m, w, w1, mu = 0.5, 0.15504447, 1700, 1000, 1, 0.083626, 0.0269564
    return (b - d * X['E'] / Ec) * (n / e ** (1/3)) + m * n / X['N']

def f_de(n, e, X):
    b, d, Ec, m, w, w1, mu = 0.5, 0.15504447, 1700, 1000, 1, 0.083626, 0.0269564
    return w * n * e ** (2/3) - w1 * n * e - d * n * e ** (2/3) * X['E'] / Ec + m * n / X['N']

def get_functions():
    return [f_n, f_ne, f_dn, f_de]

def get_function_values(functions, X, maxima):
    # Create n,e grid
    n_max, e_min, e_max = maxima
    n_vals = np.arange(1, n_max + 1, dtype=float)
    e_vals = np.linspace(e_min, e_max, num=30, dtype=float)
    n_grid, e_grid = np.meshgrid(n_vals, e_vals, indexing='ij')

    # Fill grid
    num_funcs = len(functions)
    results = np.zeros((num_funcs, len(n_vals), len(e_vals)))
    for i, func in enumerate(functions):
        results[i] = func(n_grid, e_grid, X)

    return results, e_vals