import re
import csv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from src.parametrize_transition_functions.SINDy_like_regression import do_polynomial_regression as sindy
from src.MaxEnt_inference.METimE import run_optimization as METE
from src.parametrize_transition_functions.SINDy_like_regression import get_functions, get_function_values
from src.MaxEnt_inference.empirical_BCI import make_initial_guess, check_constraints, evaluate_model, entropy, plot_RADs, single_constraint, compute_lambda_bounds, penalized_entropy, penalized_entropy_grad
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import wilcoxon
import seaborn as sns

def load_data():
    df = pd.read_csv('C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/LivingTrees19972017.csv', encoding='latin1')
    df = df.drop(columns=['SpecimenCode', 'MeasurementDate', 'Genus', 'SpeciesName', 'LifeForm', 'Family', 'Xcoord', 'Ycoord', 'Quadrat', 'Stem status'])

    # For Stem ID, remove any (A) or (B) or (C) or ... and if any duplicates arise within the same year, take the mean DBH
    df['Stem ID'] = df['Stem ID'].apply(lambda x: re.sub(r'\([A-Za-z]\)', '', str(x)).strip())
    aggregated_df = df.groupby(
        ['Stem ID', 'Year', 'PlotName', 'SpeciesCode'],
        as_index=False
    ).agg({
        'DBH': 'mean'
    }).reset_index(drop=True)

    # Calculate the minimum DBH for each PlotName
    min_dbh_per_plot = df.groupby('PlotName')['DBH'].transform('min')

    # Compute the metabolic rate: e = (DBH / min(DBH))^2
    df['e'] = (df['DBH'] / min_dbh_per_plot) ** 2

    # Calculate the number of trees per species per year per PlotName
    n = df.groupby(['SpeciesCode', 'Year', 'PlotName']).size().reset_index(name='n')

    # Map the counts back to the original DataFrame
    df = df.merge(
        n,
        on=['SpeciesCode', 'Year', 'PlotName'],
        how='left'
    )

    # Calculate N_t: Total number of individuals per Year and PlotName
    N_t = df.groupby(['Year', 'PlotName']).size().reset_index(name='N_t')

    # Calculate S_t: Total number of unique species per Year and PlotName
    S_t = df.groupby(['Year', 'PlotName'])['SpeciesCode'].nunique().reset_index(name='S_t')

    # Calculate E_t: Total metabolic rate per Year and PlotName
    E_t = df.groupby(['Year', 'PlotName'])['e'].sum().reset_index(name='E_t')

    # Merge all three back into the original DataFrame
    df = df.merge(N_t, on=['Year', 'PlotName'], how='left')
    df = df.merge(S_t, on=['Year', 'PlotName'], how='left')
    df = df.merge(E_t, on=['Year', 'PlotName'], how='left')

    # Add the values of n at the next year
    df_next = df.copy()
    df_next['Year'] = df_next['Year'] - 1
    df_next.rename(columns={'n': 'next_n'}, inplace=True)
    df_next.rename(columns={'e': 'next_e'}, inplace=True)
    df_next.rename(columns={'S_t': 'next_S'}, inplace=True)
    df_next.rename(columns={'N_t': 'next_N'}, inplace=True)
    df_next.rename(columns={'E_t': 'next_E'}, inplace=True)
    df_next = df_next[['SpeciesCode', 'Year', 'Stem ID', 'PlotName', 'next_n', 'next_e', 'next_S', 'next_N', 'next_E']]
    df = df.merge(df_next, how='left', on=['SpeciesCode', 'Year', 'Stem ID', 'PlotName'])

    # Change NaNs in df['next_n'] and df['next_e'] to 0'
    df['next_n'] = df['next_n'].fillna(0).astype(int)
    df['next_e'] = df['next_e'].fillna(0).astype(int)

    # Fill in S, N, and E by any other entry from the same census
    df[['next_S', 'next_N', 'next_E']] = df.groupby(['SpeciesCode', 'Year', 'PlotName'])[['next_S', 'next_N', 'next_E']].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    df['dn'] = df['next_n'] - df['n']
    df['de'] = df['next_e'] - df['e']
    df['dS'] = df['next_S'] - df['S_t']
    df['dN/S'] = (df['next_N'] - df['N_t']) / df['S_t']
    df['dE/S'] = (df['next_E'] - df['E_t']) / df['S_t']

    df = df.drop(columns=['next_n', 'next_S', 'next_e', 'next_N', 'next_E'], axis=1)
    df = df.dropna(how='any')
    df = df.drop(columns=['DBH'])

    # Rename columns
    df = df.rename(columns={'Year': 'census',
                            'SpeciesCode': 'species',
                            'Stem ID': 'TreeID'})

    # plot all the time series of n and e
    #make_individual_plots(df)
    return df

def get_empirical_RAD(df):
    df = df[['species', 'n']].drop_duplicates()

    # Create rank abundance distribution
    df = df.sort_values(by='n', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    rad = df[['rank', 'n']].rename(columns={'n': 'abundance'})

    return rad

def add_row(data, plot):
    filename = f"costa_rica_{plot}_results_per_slack_weight.csv"
    file_exists = os.path.isfile(filename)
    columns = ["census", "PlotName", "slack_weight", "AIC", "RMSE", "MAE", "entropy"]

    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(data)

def run_optimization(vars, macro_var, func_vals, slack_weight=1, maxiter=1e08):
    vars = np.asarray(vars, dtype=float)

    # Compute bounds (to prevent overflow in exp)
    f3_vals = func_vals[2, :, :]
    min_f3, max_f3 = f3_vals.min(), f3_vals.max()
    bounds_dn = compute_lambda_bounds(min_f3, max_f3, 100)

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
            (0, 18) / scales[0],
            (0, 18) / scales[1],
            bounds_dn / scales[2],
            bounds_de / scales[3]
    ]

    constraint_order = ['N/S', 'E/S', 'dN/S', 'dE/S']

    constraints = [{
        'type': 'eq',
        'fun': lambda vars, F_k=macro_var[name], idx=i:
        single_constraint(vars, func_vals, idx, F_k, scales)
    } for i, name in enumerate(constraint_order[:4])]


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

def make_individual_plots(df):
    plots = df['PlotName'].unique()
    for plot in plots:
        plot_df = df[df['PlotName'] == plot]
        species = plot_df['species'].unique()

        plt.figure(figsize=(10, 6))
        for sp in species:
            sp_df = plot_df[plot_df['species'] == sp].sort_values(by='census')
            plt.plot(sp_df['census'], sp_df['n'])

        plt.xlabel('Year')
        plt.ylabel('n')
        plt.title(plot)
        plt.tight_layout()
        plt.show()

def select_best_slack_weight(df, metric="MAE"):
    results = []

    for census in df['census'].unique():
        # filter by both quad and census
        df_subset = df[df['census'] == census]

        # find row that minimizes MAE
        best_idx = df_subset[f'METimE_{metric}'].idxmin()
        best_row = df_subset.loc[best_idx]

        results.append(best_row)

    # return a DataFrame of the selected best rows
    return pd.DataFrame(results).reset_index(drop=True)

def runPlotSplit(df, plot="LSUR", split=2004):
    ####################################
    ### Start looping over all plots ###
    ####################################

    # for plot in pd.unique(df['PlotName']):

    # Create list to store results
    results_list = []

    for label in [f"before{split}", f"after{split}"]:
        df = load_data()
        input = df[df['PlotName'] == plot]
        input = input.drop(columns=['PlotName'])

        if label == f"before{split}":
            input = input[input['census'] < split]
        else:
            input = input[input['census'] >= split]

        # Compute polynomial coefficients
        alphas, r2_dn, betas, r2_de, scaler = sindy(input, lv_ratio=0.0, outlier_removal=True)
        functions = get_functions()
        r2s = pd.DataFrame({'r2_dn': [r2_dn], 'r2_de': [r2_de]})
        r2s.to_csv(f'costa_rica_r2_tf{plot}.csv', index=False)
        alphas = alphas['Coefficient'].values
        betas = betas['Coefficient'].values

        years = input['census'].unique()
        del df, input

        for census in years:
            print(f"\n Census: {census} \n")

            df = load_data()
            input_census = df[(df['PlotName'] == plot) & (df['census'] == census)]

            del df

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

            empirical_rad = get_empirical_RAD(input_census)['abundance']

            # Precompute functions(n, e)
            # max_n = int(min(X['N_t'], 1.5 * max(input_census['n'])))
            max_n = int(X['N_t'] - X['S_t'])

            if max_n > 1000:
                max_n = int(max(input_census['n']))
            min_e = max(1, -1.5 * input_census['e'].quantile(0.15))
            max_e = min(X['E_t'], 1.5 * input_census['e'].quantile(0.85))

            func_vals, _ = get_function_values(functions, X, alphas, betas, scaler,
                                               [max_n, min_e, max_e],
                                               show_landscape=False)

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
                maxiter=1e10
            )
            METE_lambdas = np.append(METE_lambdas, [0, 0])
            mete_constraint_errors = check_constraints(METE_lambdas, input_census, func_vals)
            METE_results, METE_rad = evaluate_model(METE_lambdas, X, func_vals, empirical_rad, mete_constraint_errors)
            # METE_lambdas = np.append(METE_lambdas, [0, 0])

            #######################################
            #####           METimE            #####
            #######################################
            prev_best_MAE = np.inf
            for w_exp in np.arange(-2, 2, 1, dtype=float):
            #for w_exp in [-1, 0, 1]:
                w_base_list = [1, 5, 9]

                for w_base in w_base_list:
                    w = w_base * 10 ** w_exp
                    print(" ")
                    print("----------METimE----------")
                    METimE_lambdas = run_optimization(METE_lambdas, macro_var, func_vals, slack_weight=w, maxiter=1e10)
                    print("Optimized lambdas: {}".format(METimE_lambdas[:4]))
                    # print("Slack variables: {}".format(METimE_lambdas[4:]))
                    metime_constraint_errors = check_constraints(METimE_lambdas, input_census, func_vals)
                    METimE_results, METimE_rad = evaluate_model(METimE_lambdas, X, func_vals, empirical_rad,
                                                                metime_constraint_errors)
                    print(
                        f"AIC: {METimE_results['AIC'].values[0]}, RMSE: {METimE_results['RMSE'].values[0]}, MAE: {METimE_results['MAE'].values[0]}")

                    ##########################################
                    #####           Save results         #####
                    ##########################################
                    results_list.append({
                        'PlotName': plot,
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

                    add_row({
                        "census": census,
                        "PlotName": plot,
                        "slack_weight": w,
                        "AIC": METimE_results['AIC'].values[0],
                        "RMSE": METimE_results['RMSE'].values[0],
                        "MAE": METimE_results['MAE'].values[0],
                        "entropy": -entropy(METimE_lambdas[:4], func_vals)
                    }, plot)

                    if METimE_results['MAE'].values[0] < prev_best_MAE:
                        plot_RADs(empirical_rad, METE_rad, METimE_rad, f'plot_{plot}_census_{census}_split={split}',
                                  'Empirical',
                                  weight={w}, use_log=True)
                        prev_best_MAE = METimE_results['MAE'].values[0]

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'costa_rica_df{plot}_split={split}.csv', index=False)

def do_statistics(df_model, metric="MAE"):
    results = []  # Store test results

    wilcoxon_res = wilcoxon(df_model[f'METE_{metric}'], df_model[f'METimE_{metric}'], method="asymptotic")
    p_val_wilcoxon = wilcoxon_res.pvalue
    z_val_wilcoxon = wilcoxon_res.zstatistic

    # Collect results
    results.append({
        'wilcoxon_p': p_val_wilcoxon,
        'wilcoxon_z': z_val_wilcoxon,
        'median_METE': np.median(df_model[f'METE_{metric}']),
        'median_METimE': np.median(df_model[f'METimE_{metric}']),
    })

    results_df = pd.DataFrame(results)

    results_df = results_df[['median_METE', 'median_METimE', 'wilcoxon_p', 'wilcoxon_z']]

    results_df['significant'] = results_df['wilcoxon_p'].apply(
        lambda p: 'yes' if p < 0.05 else 'no'
    )

    results_df['effect_size'] = results_df['wilcoxon_z'].apply(
        lambda x: x / np.sqrt(len(df_model) * 2)
    )

    results_df['effect_category'] = results_df['effect_size'].apply(
        lambda p: 'none' if np.abs(p) < 0.1 else
        'small' if np.abs(p) < .3 else
        'medium' if np.abs(p) < .5 else
        'large'
    )

    return results_df

def do_clustering(df):
    # Pivot to aggregate by species (mean across years)
    pivot_df = df.pivot_table(index='species', values=['n', 'e', 'dn', 'de'], aggfunc='mean')
    features = pivot_df.values
    species = pivot_df.index

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Cluster (e.g., 3 clusters)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Visualize with PCA
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap='viridis')
    for i, species_name in enumerate(species):
        plt.text(pca_features[i, 0], pca_features[i, 1], species_name, fontsize=12)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Species Clusters')
    plt.show()

    # Add cluster labels to the original data
    pivot_df['cluster'] = clusters
    print(pivot_df)

    return pivot_df

if __name__ == '__main__':

    #########################
    ### Load & clean data ###
    #########################

    df = load_data()

    #runPlotSplit(df, plot="JE", split=2010)

    path = "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/PythonProjects/METimE_2026/METimE_2026/src/MaxEnt_inference/costa_rica_dfJE_split=2010.csv"
    df_plot = pd.read_csv(path)
    df = select_best_slack_weight(df_plot, "MAE")

    x_median = np.median(df[f"METE_MAE"])
    y_median = np.median(df[f"METimE_MAE"])

    stats_df = do_statistics(df, metric="MAE")

    unique_census = df['census'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_census)))
    palette = dict(zip(unique_census, colors))

    sns.scatterplot(
        data=df,
        x=f"METE_MAE",
        y=f"METimE_MAE",
        hue="census",
        palette=palette,
        alpha=0.5,
        zorder=1
    )

    # Plot median cross
    plt.scatter(x_median, y_median, s=300, marker='X', edgecolors='black', linewidth=2, zorder=2)

    # Diagonal x=y line
    lims = [
        np.nanmin([plt.xlim(), plt.ylim()]),
        np.nanmax([plt.xlim(), plt.ylim()])
    ]
    plt.plot(lims, lims, '--', color="black", alpha=.7, lw=1.5, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)

    # Styling
    plt.xlabel("METE", fontsize=12)
    plt.ylabel("METimE", fontsize=12)
    plt.title("MAE", fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=10)

    plt.show()