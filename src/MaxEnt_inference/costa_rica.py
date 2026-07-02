import pandas as pd
import numpy as np
import gc
import re
import csv
import os

from matplotlib import pyplot as plt

from src.parametrize_transition_functions.SINDy_like_regression import do_polynomial_regression as sindy
from src.MaxEnt_inference.METimE import run_optimization as METE
from src.parametrize_transition_functions.SINDy_like_regression import get_functions, get_function_values
from src.MaxEnt_inference.empirical_BCI import make_initial_guess, check_constraints, evaluate_model, plot_RADs, single_constraint, compute_lambda_bounds, penalized_entropy, penalized_entropy_grad
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

def get_empirical_RAD(df, census):
    df = df[df['census'] == census]
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

def add_row_results_list(data, plot):
    filename = f"C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/costa_rica/results_{plot}.csv"
    file_exists = os.path.isfile(filename)
    columns = [
    'PlotName',
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
            (-18, 18) / scales[0],
            (-18, 18) / scales[1],
            bounds_dn / scales[2],
            bounds_de / scales[3]
    ]

    constraint_order = ['N/S', 'E/S', 'dN/S', 'dE/S']

    constraints = [{
        'type': 'eq',
        'fun': lambda vars, F_k=macro_var[name], idx=i:
        single_constraint(vars, func_vals, idx, F_k, scales)
    } for i, name in enumerate(constraint_order[:2])]


    result = minimize(penalized_entropy,
                      vars,
                      jac=penalized_entropy_grad,
                      args=(func_vals, macro_var, scales, slack_weight),
                      constraints=constraints,
                      bounds=bounds,
                      method="trust-constr",
                      options={'maxiter':maxiter,
                               'initial_tr_radius': 0.05,
                               'xtol': 1e-12,
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


if __name__ == '__main__':

    #########################
    ### Load & clean data ###
    #########################

    # df = pd.read_csv('C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/LivingTrees19972017.csv', encoding='latin1')
    # df = df.drop(columns=['SpecimenCode', 'MeasurementDate', 'Genus', 'SpeciesName', 'LifeForm', 'Family', 'Xcoord', 'Ycoord', 'Quadrat', 'Stem status'])
    #
    # # For Stem ID, remove any (A) or (B) or (C) or ... and if any duplicates arise within the same year, take the mean DBH
    # df['Stem ID'] = df['Stem ID'].apply(lambda x: re.sub(r'\([A-Za-z]\)', '', str(x)).strip())
    # aggregated_df = df.groupby(
    #     ['Stem ID', 'Year', 'PlotName', 'SpeciesCode'],
    #     as_index=False
    # ).agg({
    #     'DBH': 'mean'
    # }).reset_index(drop=True)
    #
    # # Calculate the minimum DBH for each PlotName
    # min_dbh_per_plot = df.groupby('PlotName')['DBH'].transform('min')
    #
    # # Compute the metabolic rate: e = (DBH / min(DBH))^2
    # df['e'] = (df['DBH'] / min_dbh_per_plot) ** 2
    #
    # # Calculate the number of trees per species per year per PlotName
    # n = df.groupby(['SpeciesCode', 'Year', 'PlotName']).size().reset_index(name='n')
    #
    # # Map the counts back to the original DataFrame
    # df = df.merge(
    #     n,
    #     on=['SpeciesCode', 'Year', 'PlotName'],
    #     how='left'
    # )
    #
    # # Calculate N_t: Total number of individuals per Year and PlotName
    # N_t = df.groupby(['Year', 'PlotName']).size().reset_index(name='N_t')
    #
    # # Calculate S_t: Total number of unique species per Year and PlotName
    # S_t = df.groupby(['Year', 'PlotName'])['SpeciesCode'].nunique().reset_index(name='S_t')
    #
    # # Calculate E_t: Total metabolic rate per Year and PlotName
    # E_t = df.groupby(['Year', 'PlotName'])['e'].sum().reset_index(name='E_t')
    #
    # # Merge all three back into the original DataFrame
    # df = df.merge(N_t, on=['Year', 'PlotName'], how='left')
    # df = df.merge(S_t, on=['Year', 'PlotName'], how='left')
    # df = df.merge(E_t, on=['Year', 'PlotName'], how='left')
    #
    # # Add the values of n at the next year
    # df_next = df.copy()
    # df_next['Year'] = df_next['Year'] - 1
    # df_next.rename(columns={'n': 'next_n'}, inplace=True)
    # df_next.rename(columns={'e': 'next_e'}, inplace=True)
    # df_next.rename(columns={'S_t': 'next_S'}, inplace=True)
    # df_next.rename(columns={'N_t': 'next_N'}, inplace=True)
    # df_next.rename(columns={'E_t': 'next_E'}, inplace=True)
    # df_next = df_next[['SpeciesCode', 'Year', 'Stem ID', 'PlotName', 'next_n', 'next_e', 'next_S', 'next_N', 'next_E']]
    # df = df.merge(df_next, how='left', on=['SpeciesCode', 'Year', 'Stem ID', 'PlotName'])
    #
    # # Change NaNs in df['next_n'] and df['next_e'] to 0'
    # df['next_n'] = df['next_n'].fillna(0).astype(int)
    # df['next_e'] = df['next_e'].fillna(0).astype(int)
    #
    # # Fill in S, N, and E by any other entry from the same census
    # df[['next_S', 'next_N', 'next_E']] = df.groupby(['SpeciesCode', 'Year', 'PlotName'])[['next_S', 'next_N', 'next_E']].transform(
    #     lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    #
    # df['dn'] = df['next_n'] - df['n']
    # df['de'] = df['next_e'] - df['e']
    # df['dS'] = df['next_S'] - df['S_t']
    # df['dN/S'] = (df['next_N'] - df['N_t']) / df['S_t']
    # df['dE/S'] = (df['next_E'] - df['E_t']) / df['S_t']
    #
    # df = df.drop(columns=['next_n', 'next_S', 'next_e', 'next_N', 'next_E'], axis=1)
    # df = df.dropna(how='any')
    # df = df.drop(columns=['DBH'])
    #
    # # Rename columns
    # df = df.rename(columns={'Year': 'census',
    #                         'SpeciesCode': 'species',
    #                         'Stem ID': 'TreeID'})
    #
    # # plot all the time series of n and e
    # make_individual_plots(df)
    #
    # del E_t, N_t, S_t, aggregated_df, df_next, min_dbh_per_plot, n
    #
    # # After all your processing, before deleting variables:
    # unique_plots = df['PlotName'].unique()
    #
    # # Save a CSV for each PlotName
    # for plot in unique_plots:
    #     plot_df = df[df['PlotName'] == plot]
    #     plot_df.to_csv(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/CostaRicaPerPlot/CostaRicaTrees_{plot}.csv', index=False)

    ####################################
    ### Start looping over all plots ###
    ####################################

    for plot in ['TIR', 'JE', 'LSUR', 'CR', 'LEP', 'BEJ', 'LEPviejo', 'SV']:
        print("Plot: ", plot)
        input = pd.read_csv(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/CostaRicaPerPlot/CostaRicaTrees_{plot}.csv')
        input = input.drop(columns=['PlotName'])

        # Compute polynomial coefficients
        best_r2_dn, best_r2_de, best_alphas, best_betas = -np.inf, -np.inf, [], []
        for lv_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

            alphas, r2_dn, betas, r2_de, scaler = sindy(input, lv_ratio=lv_ratio, outlier_removal=True)

            if r2_dn > best_r2_dn:
                best_r2_dn = r2_dn
                best_alphas = alphas

            if r2_de > best_r2_de:
                best_r2_de = r2_de
                best_betas = betas

        print(f"Best r2_dn: {best_r2_dn}, Best r2_de: {best_r2_de}")
        print(f"Best alphas: {best_alphas['Coefficient'].values}, Best betas: {best_betas['Coefficient'].values}")

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
            empirical_rad = get_empirical_RAD(input, census)['abundance']

            # Precompute functions(n, e)
            # max_n = int(min(X['N_t'], 1.5 * max(input_census['n'])))
            max_n = int(X['N_t'] - X['S_t'])

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

            #######################################
            #####           METimE            #####
            #######################################
            # Delete everything but macro_var and func_vals
            prev_best_MAE = np.inf
            for w in [100, 10, 1, 0.01, 0]:
                print(" ")
                print("----------METimE----------")
                METimE_lambdas = run_optimization(METE_lambdas, macro_var, func_vals, slack_weight=w, maxiter=1e5)
                print("Optimized lambdas: {}".format(METimE_lambdas[:4]))
                    # print("Slack variables: {}".format(METimE_lambdas[4:]))
                metime_constraint_errors = check_constraints(METimE_lambdas, input_census, func_vals)
                METimE_results, METimE_rad = evaluate_model(METimE_lambdas, X, func_vals, empirical_rad,
                                                                metime_constraint_errors)
                print(
                    f"AIC: {METimE_results['AIC'].values[0]}, RMSE: {METimE_results['RMSE'].values[0]}, MAE: {METimE_results['MAE'].values[0]}")


                add_row_results_list(
                        {
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
                        },
                        plot
                    )

                # add_row({
                #         "census": census,
                #         "PlotName": plot,
                #         "slack_weight": w,
                #         "AIC": METimE_results['AIC'].values[0],
                #         "RMSE": METimE_results['RMSE'].values[0],
                #         "MAE": METimE_results['MAE'].values[0],
                #         "entropy": -entropy(METimE_lambdas[:4], func_vals)
                #     }, plot)

                if METimE_results['MAE'].values[0] < prev_best_MAE:
                    plot_RADs(empirical_rad, METE_rad, METimE_rad, f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/costa_rica/plot_{plot}_census_{census}.png', 'Empirical',
                                  weight={w}, use_log=True)
                    prev_best_MAE = METimE_results['MAE'].values[0]

            gc.collect()