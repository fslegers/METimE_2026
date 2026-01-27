import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNetCV, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def remove_outliers(df):
    df_clean = df.copy()

    for col in ['n', 'e', 'dn', 'de']:
        Q1 = np.percentile(df_clean[col], 15, method='midpoint')
        Q3 = np.percentile(df_clean[col], 85, method='midpoint')
        IQR = Q3 - Q1

        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR

        # Keep only values within the bounds
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean

def do_polynomial_regression(df, lv_ratio=0.6):
    # Make base nonlinear transformations of e and n
    df = df.copy()

    with open('variance simulated BCI.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            df['dn'].tolist(),
            df['de'].tolist()
        ])

    # Remove outliers
    df = remove_outliers(df)

    # print mean and variance in df['de']
    # Protect against zero/negative values for logs and inverses
    eps = 1e-12
    e = df['e'].clip(lower=eps)

    # Custom features
    df['(e ** (-1))'] = 1.0 / e
    df['(e ** (-3/4))'] = e ** (-3 / 4)
    df['(e ** (-2/3))'] = e ** (-2 / 3)
    df['(e ** (-1/2))'] = e ** (-1 / 2)
    df['(e ** (-1/4))'] = e ** (-1 / 4)
    df['(e ** (1/4))'] = e ** (1 / 4)
    df['(e ** (1/2))'] = e ** (1 / 2)
    df['(e ** (2/3))'] = e ** (2 / 3)
    df['(e ** (3/4))'] = e ** (3 / 4)
    df['(e)'] = e
    df['(e ** (3/2))'] = e ** (3 / 2)

    df['(n ** (1/2))'] = df['n'] ** (1 / 2)
    df['(n)'] = df['n']
    df['(n ** (3/4))'] = df['n'] ** (3 / 4)
    df['(n ** (3/2))'] = df['n'] ** (3 / 2)

    df['(np.log(n))'] = np.log(df['n'])
    df['(np.log(e))'] = np.log(e)

    df['(1/N_t)'] = 1.0 / df['N_t']
    df['(1/E_t)'] = 1.0 / df['E_t']

    df.rename(columns={'N_t': '(N_t)', 'E_t': '(E_t)', 'S_t': '(S_t)'}, inplace=True)

    # These are the columns that will be used to create polynomial features
    poly_cols = ['(e ** (-1))', '(e ** (-3/4))', '(e ** (-2/3))', '(e ** (-1/2))', '(e ** (-1/4))', '(e ** (1/4))',
                 '(e ** (1/2))', '(e ** (2/3))', '(e ** (3/4))', '(e)', '(e ** (3/2))',
                 '(n ** (1/2))', '(n)', '(n ** (3/4))', '(n ** (3/2))',
                 '(np.log(n))',
                 '(np.log(e))',
                 '(N_t)', '(1/N_t)',
                 '(E_t)', '(1/E_t)']

    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=True)
    poly_features = poly.fit_transform(df[poly_cols])

    # Create a new DataFrame with polynomial features
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    # Concatenate polynomial features back to the original DataFrame
    df = pd.concat([df.drop(columns=poly_cols), poly_df], axis=1)

    # Drop 'tree_id' and dN/S and dE/S columns
    df = df.drop(columns=['TreeID', 'dN/S', 'dE/S', 'dS'], errors='ignore')

    df_means = df.groupby(['census', 'species']).mean().reset_index()
    dn_obs = df_means['dn'].values
    de_obs = df_means['de'].values
    targets = [(dn_obs, 'dn'), (de_obs, 'de')]

    # Here, we take averages per species over features calculated per tree
    # Group by (t, species_id) and sum all features
    df_grouped = df.groupby(['census', 'species']).sum().reset_index()

    # Run STLSQ
    coef_dn, r2_dn, coef_de, r2_de, scaler = stepwise_sparse_regression(df_grouped, targets, lv_ratio)
    print(f"R2 dn: {r2_dn}, R2 de: {r2_de}")

    return coef_dn, r2_dn, coef_de, r2_de, scaler

def stepwise_sparse_regression(df, targets, alpha=0.01):
    """
    Sequential Threshold Least Squares (STLSQ) for sparse regression.
    Fits dn and de simultaneously, returning coefficient DataFrames.

    Args:
        df_grouped : DataFrame
            Must include columns ['dn','de'] plus polynomial features.
        lambda_reg : float
            Threshold below which coefficients are zeroed.
        max_iter : int
            Maximum number of iterations.

    Returns:
        coef_dn : pd.DataFrame with columns ['feature','coefficient']
        coef_de : pd.DataFrame with columns ['feature','coefficient']
    """

    # Split into target and features
    X = df.drop(columns=['census', 'species', 'dn', 'de'], errors='ignore')

    feature_names = X.columns.tolist()

    # Standardize features
    scaler = StandardScaler(with_mean=False) # TODO: think about this!
    X_scaled = scaler.fit_transform(X)

    results = []
    r2s = []

    for y_obs, target_name in targets:
        # initial Lasso fit
        model = Lasso(alpha=alpha, fit_intercept=False)
        model.fit(X_scaled, y_obs)
        y_pred = model.predict(X_scaled)
        prev_r2 = r2_score(y_obs, y_pred)
        best_coef = model.coef_.copy()

        if np.all(best_coef == 0):
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'Coefficient': best_coef.tolist()
            })
            results.append(coef_df)
            r2s.append(r2_score(y_obs, y_pred))
            continue

        for _ in range(len(feature_names) - 2):
            # Identify the index of the smallest non-zero coefficient
            non_zero_idx = np.where(best_coef != 0)[0]
            if len(non_zero_idx) <= 1:
                break  # all coefficients eliminated

            smallest_idx = non_zero_idx[np.argmin(np.abs(best_coef[non_zero_idx]))]

            # Zero out the smallest coefficient
            new_coef = best_coef.copy()
            new_coef[smallest_idx] = 0

            # Refit only on remaining non-zero features
            mask = new_coef != 0
            if not np.any(mask):
                break  # nothing left to fit

            model = Lasso(alpha=alpha, fit_intercept=False)
            model.fit(X_scaled[:, mask], y_obs)

            # Update coefficients
            new_coef = np.zeros_like(new_coef)
            new_coef[mask] = model.coef_

            # compute R²
            y_pred = model.predict(X_scaled[:, mask])
            r2 = r2_score(y_obs, y_pred)

            # stop if prediction accuracy decreases too much
            if prev_r2 - r2 > 1e-4:
                break

            prev_r2 = r2
            best_coef = new_coef.copy()

        # Recalculate non-zero coefficients (with weaker l1 norm regularization)
        mask = best_coef != 0

        if np.any(mask):
            model = ElasticNetCV(l1_ratio=0.5, alphas=np.logspace(-3, 1, 50), fit_intercept=False)
            model.fit(X_scaled[:, mask], y_obs)
            coef_new = np.zeros_like(best_coef)
            coef_new[mask] = model.coef_
            best_coef = coef_new
            y_pred = model.predict(X_scaled[:, mask])
            r2 = r2_score(y_obs, y_pred)
        else:
            r2 = 0.0

        # store results
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'Coefficient': best_coef.tolist()
        })
        results.append(coef_df)
        r2s.append(r2)

    coef_dn, coef_de = results
    r2_dn, r2_de = r2s

    return coef_dn, r2_dn, coef_de, r2_de, scaler

def f_n(n, e, X, alphas, betas, scaler):
    return n

def f_ne(n, e, X, alphas, betas, scaler):
    return n * e

def f_dn(n, e, X, alphas, betas, scaler):
    func_vals = get_scaled_features(n, e, X, alphas, scaler)
    return np.clip(func_vals, -n, None)

def f_de(n, e, X, alphas, betas, scaler):
    func_vals = get_scaled_features(n, e, X, betas, scaler)
    return np.clip(func_vals, -e, None)

def get_functions():
    return [f_n, f_ne, f_dn, f_de]

def get_function_values(functions, X, alphas, betas, scaler, maxima, show_landscape=False, training_points=None):
    # Create n,e grid
    n_max, e_min, e_max = maxima
    n_vals = np.arange(1, n_max + 1, dtype=float)
    e_vals = np.linspace(e_min, e_max, num=30, dtype=float)
    n_grid, e_grid = np.meshgrid(n_vals, e_vals, indexing='ij')

    # Fill grid
    num_funcs = len(functions)
    results = np.zeros((num_funcs, len(n_vals), len(e_vals)))
    for i, func in enumerate(functions):
        results[i] = func(n_grid, e_grid, X, alphas, betas, scaler)

    # Assume results[i] is already computed for all i
    if show_landscape:
        fig, axes = plt.subplots(1, num_funcs, subplot_kw={"projection": "3d"},
                                 figsize=(6 * num_funcs, 10))

        function_names = [r"n", r"n e", r"f(n,e)", r"h(n,e)"]

        for i in range(num_funcs):
            # --- Global plot ---
            ax_global = axes[i]
            surf = ax_global.plot_surface(
                n_grid, e_grid, results[i],
                cmap="viridis", edgecolor="none", alpha=0.9
            )
            ax_global.set_title(function_names[i])
            ax_global.set_xlabel("n")
            ax_global.set_ylabel("e")
            ax_global.set_zlabel("f(n,e)")
            fig.colorbar(surf, ax=ax_global, shrink=0.4, aspect=10)

            # # --- Zoomed plot ---
            # ax_zoom = axes[1, i]
            # if training_points is not None:
            #     tp = np.array(training_points)
            #     n_min, n_max = tp[:, 0].min(), tp[:, 0].max()
            #     e_min, e_max = tp[:, 1].min(), tp[:, 1].max()
            #     n_pad = 0.1 * (n_max - n_min)
            #     e_pad = 0.1 * (e_max - e_min)
            #     n_min, n_max = n_min - n_pad, n_max + n_pad
            #     e_min, e_max = e_min - e_pad, e_max + e_pad
            #
            #     # Mask grid to zoomed region
            #     mask = ((n_grid >= n_min) & (n_grid <= n_max) &
            #             (e_grid >= e_min) & (e_grid <= e_max))
            #     n_zoom = np.where(mask, n_grid, np.nan)
            #     e_zoom = np.where(mask, e_grid, np.nan)
            #     z_zoom = np.where(mask, results[i], np.nan)
            # else:
            #     n_zoom, e_zoom, z_zoom = n_grid, e_grid, results[i]
            #
            # surf_zoom = ax_zoom.plot_surface(
            #     n_zoom, e_zoom, z_zoom,
            #     cmap="viridis", edgecolor="none", alpha=0.9
            # )
            # ax_zoom.set_xlim(n_grid.min(), n_grid.max())
            # ax_zoom.set_ylim(e_grid.min(), e_grid.max())
            # ax_zoom.set_title(f"Function {i + 1} (Zoom)")
            # ax_zoom.set_xlabel("n")
            # ax_zoom.set_ylabel("e")
            # ax_zoom.set_zlabel("f(n,e)")
            # fig.colorbar(surf_zoom, ax=ax_zoom, shrink=0.5, aspect=10)

        plt.tight_layout()
        plt.show()

    return results, e_vals

def get_raw_features(n, e, X):
    Nt, Et, St = X['N_t'] * np.ones_like(n), X['E_t'] * np.ones_like(n), X['S_t'] * np.ones_like(n)
    logn, loge = np.log(n), np.log(e)

    features = [
        e, St, n,
        np.ones_like(n), (e ** (-1)), (e ** (-3 / 4)), (e ** (-2 / 3)),
         (e ** (-1 / 2)), (e ** (-1 / 4)), (e ** (1 / 4)), (e ** (1 / 2)),
         (e ** (2 / 3)), (e ** (3 / 4)), (e), (e ** (3 / 2)),
         (n ** (1 / 2)), (n), (n ** (3 / 4)), (n ** (3 / 2)),
         (logn), (loge), (Nt), (1 / Nt), (Et),
         (1 / Et), (e ** (-1)) ** 2, (e ** (-1)) * (e ** (-3 / 4)),
         (e ** (-1)) * (e ** (-2 / 3)), (e ** (-1)) * (e ** (-1 / 2)),
         (e ** (-1)) * (e ** (-1 / 4)), (e ** (-1)) * (e ** (1 / 4)),
         (e ** (-1)) * (e ** (1 / 2)), (e ** (-1)) * (e ** (2 / 3)),
         (e ** (-1)) * (e ** (3 / 4)), (e ** (-1)) * (e),
         (e ** (-1)) * (e ** (3 / 2)), (e ** (-1)) * (n ** (1 / 2)),
         (e ** (-1)) * (n), (e ** (-1)) * (n ** (3 / 4)),
         (e ** (-1)) * (n ** (3 / 2)), (e ** (-1)) * (logn),
         (e ** (-1)) * (loge), (e ** (-1)) * (Nt),
         (e ** (-1)) * (1 / Nt), (e ** (-1)) * (Et), (e ** (-1)) * (1 / Et),
         (e ** (-3 / 4)) ** 2, (e ** (-3 / 4)) * (e ** (-2 / 3)),
         (e ** (-3 / 4)) * (e ** (-1 / 2)), (e ** (-3 / 4)) * (e ** (-1 / 4)),
         (e ** (-3 / 4)) * (e ** (1 / 4)), (e ** (-3 / 4)) * (e ** (1 / 2)),
         (e ** (-3 / 4)) * (e ** (2 / 3)), (e ** (-3 / 4)) * (e ** (3 / 4)),
         (e ** (-3 / 4)) * (e), (e ** (-3 / 4)) * (e ** (3 / 2)),
         (e ** (-3 / 4)) * (n ** (1 / 2)), (e ** (-3 / 4)) * (n),
         (e ** (-3 / 4)) * (n ** (3 / 4)), (e ** (-3 / 4)) * (n ** (3 / 2)),
         (e ** (-3 / 4)) * (logn), (e ** (-3 / 4)) * (loge),
         (e ** (-3 / 4)) * (Nt), (e ** (-3 / 4)) * (1 / Nt),
         (e ** (-3 / 4)) * (Et), (e ** (-3 / 4)) * (1 / Et), (e ** (-2 / 3)) ** 2,
         (e ** (-2 / 3)) * (e ** (-1 / 2)), (e ** (-2 / 3)) * (e ** (-1 / 4)),
         (e ** (-2 / 3)) * (e ** (1 / 4)), (e ** (-2 / 3)) * (e ** (1 / 2)),
         (e ** (-2 / 3)) * (e ** (2 / 3)), (e ** (-2 / 3)) * (e ** (3 / 4)),
         (e ** (-2 / 3)) * (e), (e ** (-2 / 3)) * (e ** (3 / 2)),
         (e ** (-2 / 3)) * (n ** (1 / 2)), (e ** (-2 / 3)) * (n),
         (e ** (-2 / 3)) * (n ** (3 / 4)), (e ** (-2 / 3)) * (n ** (3 / 2)),
         (e ** (-2 / 3)) * (logn), (e ** (-2 / 3)) * (loge),
         (e ** (-2 / 3)) * (Nt), (e ** (-2 / 3)) * (1 / Nt),
         (e ** (-2 / 3)) * (Et), (e ** (-2 / 3)) * (1 / Et), (e ** (-1 / 2)) ** 2,
         (e ** (-1 / 2)) * (e ** (-1 / 4)), (e ** (-1 / 2)) * (e ** (1 / 4)),
         (e ** (-1 / 2)) * (e ** (1 / 2)), (e ** (-1 / 2)) * (e ** (2 / 3)),
         (e ** (-1 / 2)) * (e ** (3 / 4)), (e ** (-1 / 2)) * (e),
         (e ** (-1 / 2)) * (e ** (3 / 2)), (e ** (-1 / 2)) * (n ** (1 / 2)),
         (e ** (-1 / 2)) * (n), (e ** (-1 / 2)) * (n ** (3 / 4)),
         (e ** (-1 / 2)) * (n ** (3 / 2)), (e ** (-1 / 2)) * (logn),
         (e ** (-1 / 2)) * (loge), (e ** (-1 / 2)) * (Nt),
         (e ** (-1 / 2)) * (1 / Nt), (e ** (-1 / 2)) * (Et),
         (e ** (-1 / 2)) * (1 / Et), (e ** (-1 / 4)) ** 2,
         (e ** (-1 / 4)) * (e ** (1 / 4)), (e ** (-1 / 4)) * (e ** (1 / 2)),
         (e ** (-1 / 4)) * (e ** (2 / 3)), (e ** (-1 / 4)) * (e ** (3 / 4)),
         (e ** (-1 / 4)) * (e), (e ** (-1 / 4)) * (e ** (3 / 2)),
         (e ** (-1 / 4)) * (n ** (1 / 2)), (e ** (-1 / 4)) * (n),
         (e ** (-1 / 4)) * (n ** (3 / 4)), (e ** (-1 / 4)) * (n ** (3 / 2)),
         (e ** (-1 / 4)) * (logn), (e ** (-1 / 4)) * (loge),
         (e ** (-1 / 4)) * (Nt), (e ** (-1 / 4)) * (1 / Nt),
         (e ** (-1 / 4)) * (Et), (e ** (-1 / 4)) * (1 / Et), (e ** (1 / 4)) ** 2,
         (e ** (1 / 4)) * (e ** (1 / 2)), (e ** (1 / 4)) * (e ** (2 / 3)),
         (e ** (1 / 4)) * (e ** (3 / 4)), (e ** (1 / 4)) * (e),
         (e ** (1 / 4)) * (e ** (3 / 2)), (e ** (1 / 4)) * (n ** (1 / 2)),
         (e ** (1 / 4)) * (n), (e ** (1 / 4)) * (n ** (3 / 4)),
         (e ** (1 / 4)) * (n ** (3 / 2)), (e ** (1 / 4)) * (logn),
         (e ** (1 / 4)) * (loge), (e ** (1 / 4)) * (Nt),
         (e ** (1 / 4)) * (1 / Nt), (e ** (1 / 4)) * (Et),
         (e ** (1 / 4)) * (1 / Et), (e ** (1 / 2)) ** 2,
         (e ** (1 / 2)) * (e ** (2 / 3)), (e ** (1 / 2)) * (e ** (3 / 4)),
         (e ** (1 / 2)) * (e), (e ** (1 / 2)) * (e ** (3 / 2)),
         (e ** (1 / 2)) * (n ** (1 / 2)), (e ** (1 / 2)) * (n),
         (e ** (1 / 2)) * (n ** (3 / 4)), (e ** (1 / 2)) * (n ** (3 / 2)),
         (e ** (1 / 2)) * (logn), (e ** (1 / 2)) * (loge),
         (e ** (1 / 2)) * (Nt), (e ** (1 / 2)) * (1 / Nt), (e ** (1 / 2)) * (Et),
         (e ** (1 / 2)) * (1 / Et), (e ** (2 / 3)) ** 2,
         (e ** (2 / 3)) * (e ** (3 / 4)), (e ** (2 / 3)) * (e),
         (e ** (2 / 3)) * (e ** (3 / 2)), (e ** (2 / 3)) * (n ** (1 / 2)),
         (e ** (2 / 3)) * (n), (e ** (2 / 3)) * (n ** (3 / 4)),
         (e ** (2 / 3)) * (n ** (3 / 2)), (e ** (2 / 3)) * (logn),
         (e ** (2 / 3)) * (loge), (e ** (2 / 3)) * (Nt),
         (e ** (2 / 3)) * (1 / Nt), (e ** (2 / 3)) * (Et),
         (e ** (2 / 3)) * (1 / Et), (e ** (3 / 4)) ** 2, (e ** (3 / 4)) * (e),
         (e ** (3 / 4)) * (e ** (3 / 2)), (e ** (3 / 4)) * (n ** (1 / 2)),
         (e ** (3 / 4)) * (n), (e ** (3 / 4)) * (n ** (3 / 4)),
         (e ** (3 / 4)) * (n ** (3 / 2)), (e ** (3 / 4)) * (logn),
         (e ** (3 / 4)) * (loge), (e ** (3 / 4)) * (Nt),
         (e ** (3 / 4)) * (1 / Nt), (e ** (3 / 4)) * (Et),
         (e ** (3 / 4)) * (1 / Et), (e) ** 2, (e) * (e ** (3 / 2)),
         (e) * (n ** (1 / 2)), (e) * (n), (e) * (n ** (3 / 4)),
         (e) * (n ** (3 / 2)), (e) * (logn), (e) * (loge),
         (e) * (Nt), (e) * (1 / Nt), (e) * (Et), (e) * (1 / Et),
         (e ** (3 / 2)) ** 2, (e ** (3 / 2)) * (n ** (1 / 2)), (e ** (3 / 2)) * (n),
         (e ** (3 / 2)) * (n ** (3 / 4)), (e ** (3 / 2)) * (n ** (3 / 2)),
         (e ** (3 / 2)) * (logn), (e ** (3 / 2)) * (loge),
         (e ** (3 / 2)) * (Nt), (e ** (3 / 2)) * (1 / Nt), (e ** (3 / 2)) * (Et),
         (e ** (3 / 2)) * (1 / Et), (n ** (1 / 2)) ** 2, (n ** (1 / 2)) * (n),
         (n ** (1 / 2)) * (n ** (3 / 4)), (n ** (1 / 2)) * (n ** (3 / 2)),
         (n ** (1 / 2)) * (logn), (n ** (1 / 2)) * (loge),
         (n ** (1 / 2)) * (Nt), (n ** (1 / 2)) * (1 / Nt), (n ** (1 / 2)) * (Et),
         (n ** (1 / 2)) * (1 / Et), (n) ** 2, (n) * (n ** (3 / 4)),
         (n) * (n ** (3 / 2)), (n) * (logn), (n) * (loge),
         (n) * (Nt), (n) * (1 / Nt), (n) * (Et), (n) * (1 / Et),
         (n ** (3 / 4)) ** 2, (n ** (3 / 4)) * (n ** (3 / 2)),
         (n ** (3 / 4)) * (logn), (n ** (3 / 4)) * (loge),
         (n ** (3 / 4)) * (Nt), (n ** (3 / 4)) * (1 / Nt), (n ** (3 / 4)) * (Et),
         (n ** (3 / 4)) * (1 / Et), (n ** (3 / 2)) ** 2,
         (n ** (3 / 2)) * (logn), (n ** (3 / 2)) * (loge),
         (n ** (3 / 2)) * (Nt), (n ** (3 / 2)) * (1 / Nt), (n ** (3 / 2)) * (Et),
         (n ** (3 / 2)) * (1 / Et), (logn) ** 2, (logn) * (loge),
         (logn) * (Nt), (logn) * (1 / Nt), (logn) * (Et),
         (logn) * (1 / Et), (loge) ** 2, (loge) * (Nt),
         (loge) * (1 / Nt), (loge) * (Et), (loge) * (1 / Et),
         (Nt) ** 2, (Nt) * (1 / Nt), (Nt) * (Et), (Nt) * (1 / Et),
         (1 / Nt) ** 2, (1 / Nt) * (Et), (1 / Nt) * (1 / Et), (Et) ** 2,
         (Et) * (1 / Et), (1 / Et) ** 2
    ]

    return np.stack(features, axis=-1)

def get_scaled_features(n, e, X, coefs, scaler):
    raw_features = get_raw_features(n, e, X)
    n_points = raw_features.shape[0] * raw_features.shape[1]

    # Reshape to 2D: (n_points, n_features)
    raw_features_2d = raw_features.reshape(n_points, -1)

    # Scale
    scaled_features = scaler.transform(raw_features_2d)

    # Compute values
    function_values = scaled_features @ coefs

    # Reshape back to grid shape
    return function_values.reshape(n.shape)