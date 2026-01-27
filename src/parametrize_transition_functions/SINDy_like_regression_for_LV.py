import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def remove_outliers(df):
    df_clean = df.copy()

    for col in ['n', 'dn']:
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

    with open('variance LV.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            df['dn'].tolist()
        ])

    # Remove outliers
    df = remove_outliers(df)

    df['(n ** (1/2))'] = df['n'] ** (1 / 2)
    df['(n)'] = df['n']
    df['(n ** (3/4))'] = df['n'] ** (3 / 4)
    df['(n ** (3/2))'] = df['n'] ** (3 / 2)

    df['(np.log(n))'] = np.log(df['n'])

    df['(1/N_t)'] = 1.0 / df['N_t']

    df.rename(columns={'N_t': '(N_t)', 'S_t': '(S_t)'}, inplace=True)

    # These are the columns that will be used to create polynomial features
    poly_cols = ['(n ** (1/2))', '(n)', '(n ** (3/4))', '(n ** (3/2))',
                 '(np.log(n))',
                 '(N_t)', '(1/N_t)']

    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=True)
    poly_features = poly.fit_transform(df[poly_cols])

    # Create a new DataFrame with polynomial features
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    # Concatenate polynomial features back to the original DataFrame
    df = pd.concat([df.drop(columns=poly_cols), poly_df], axis=1)

    # Drop 'tree_id' and dN/S and dE/S columns
    df = df.drop(columns=['TreeID', 'dN', 'dN/S', 'dS'], errors='ignore')

    # Run STLSQ
    coef_dn, r2_dn, scaler = stepwise_sparse_regression(df, lv_ratio)
    print(f"R2 dn: {r2_dn}")

    return coef_dn, r2_dn, scaler

def stepwise_sparse_regression(df_grouped, alpha=0.01):
    # Split into target and features
    dn_obs = df_grouped['dn'].values
    X = df_grouped.drop(columns=['census', 'species', 'dn'], errors='ignore')

    feature_names = X.columns.tolist()

    # Standardize features
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)

    results = []
    r2s = []

    for y_obs, target_name in [(dn_obs, 'dn')]:

        # initial Lasso fit
        model = LinearRegression(fit_intercept=False)
        model.fit(X_scaled, y_obs)
        y_pred = model.predict(X_scaled)
        prev_r2 = r2_score(y_obs, y_pred)
        best_coef = model.coef_.copy()

        for _ in range(len(feature_names) - 1):

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

            model = LinearRegression(fit_intercept=False)
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

        # Recalculate non-zero coefficients (without l1 norm regularization)
        mask = best_coef != 0
        model = ElasticNetCV(l1_ratio=0.5, alphas=np.logspace(-3, 1, 50), fit_intercept=False)
        model.fit(X_scaled[:, mask], y_obs)
        coef_new = np.zeros_like(best_coef)
        coef_new[mask] = model.coef_
        best_coef = coef_new
        y_pred = model.predict(X_scaled[:, mask])
        r2 = r2_score(y_obs, y_pred)

        # store results
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'Coefficient': best_coef.tolist()
        })
        results.append(coef_df)
        r2s.append(r2)

    coef_dn = results[0]
    r2_dn = r2s[0]

    return coef_dn, r2_dn, scaler

def f_n(n, X, alphas, scaler):
    return n

def f_dn(n, X, alphas, scaler):
    func_vals = get_scaled_features(n, X, alphas, scaler)
    return np.clip(func_vals, -n, None)

def get_functions():
    return [f_n, f_dn]

def get_function_values(functions, X, alphas, scaler, show_landscape=False):
    # Create n grid
    n_vals = np.arange(1, int(X['N_t']) + 1, dtype=float)  # 1D array of n values

    # Fill results
    num_funcs = len(functions)
    results = np.zeros((num_funcs, len(n_vals)))  # Now results is 2D: (num_funcs, n_vals)

    for i, func in enumerate(functions):
        # Call func with only n_vals (or n_grid if you prefer), remove e
        results[i] = func(n_vals, X, alphas, scaler)

    # Assume results[i] is already computed for all i
    if show_landscape:
        fig, axes = plt.subplots(1, num_funcs,
                                 figsize=(6 * num_funcs, 10))

        function_names = [r"n", r"f(n,e)"]

        for i in range(num_funcs):
            # --- 2D plot ---
            ax = axes[i]
            ax.plot(n_vals, results[i], color='blue', alpha=0.9)
            ax.set_title(function_names[i])
            ax.set_xlabel("n")
            ax.set_ylabel("f(n)")

        plt.tight_layout()
        plt.show()

    return results

def get_raw_features(n, X):
    Nt, St = X['N_t'] * np.ones_like(n), X['S_t'] * np.ones_like(n)
    logn = np.log(n)

    features = [
        n,
        St,
        np.ones_like(n),
        (n ** (1 / 2)),
        (n),
        (n ** (3 / 4)),
        (n ** (3 / 2)),
        (logn),
        (Nt),
        (1 / Nt),
        (n ** (1 / 2)) ** 2,
        (n ** (1 / 2)) * (n),
        (n ** (1 / 2)) * (n ** (3 / 4)),
        (n ** (1 / 2)) * (n ** (3 / 2)),
        (n ** (1 / 2)) * (logn),
        (n ** (1 / 2)) * (Nt),
        (n ** (1 / 2)) * (1 / Nt),
        (n) ** 2,
        (n) * (n ** (3 / 4)),
        (n) * (n ** (3 / 2)),
        (n) * (logn),
        (n) * (Nt),
        (n) * (1 / Nt),
        (n ** (3 / 4)) ** 2,
        (n ** (3 / 4)) * (n ** (3 / 2)),
        (n ** (3 / 4)) * (logn),
        (n ** (3 / 4)) * (Nt),
        (n ** (3 / 4)) * (1 / Nt),
        (n ** (3 / 2)) ** 2,
        (n ** (3 / 2)) * (logn),
        (n ** (3 / 2)) * (Nt),
        (n ** (3 / 2)) * (1 / Nt),
        (logn) ** 2,
        (logn) * (Nt),
        (logn) * (1 / Nt),
        (Nt) ** 2,
        (Nt) * (1 / Nt),
        (1 / Nt) ** 2
    ]

    return np.stack(features, axis=-1)

def get_scaled_features(n, X, coefs, scaler):
    raw_features = get_raw_features(n, X)
    #n_points = raw_features.shape[0] * raw_features.shape[1]

    # Reshape to 2D: (n_points, n_features)
    #raw_features_2d = raw_features.reshape(n_points, -1)

    # Scale
    scaled_features = scaler.transform(raw_features)

    # Compute values
    function_values = scaled_features @ coefs

    # Reshape back to grid shape
    return function_values.reshape(n.shape)

