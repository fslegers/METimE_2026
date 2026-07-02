import glob
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
import seaborn as sns


def remove_outliers(df, cols):
    """Removes rows from df where any of the specified columns have outliers using the IQR method."""
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
        df = df[mask]
    return df


def spearman_summary(x, y, x_name, y_name):
    rho, pval = spearmanr(x, y)
    print(f"Spearman correlation between '{x_name}' and '{y_name}':")
    print(f"  ρ = {rho:.4f}, p = {pval:.4e}\n")
    return rho, pval


def scatter_and_spearman(df, x_col, y_col, x_label=None, y_label=None,
                         alpha=0.1, xlim=None, ylim=None,
                         color_split_y=False, above_color='blue', below_color='red'):
    """
    Removes outliers, creates a scatter plot (with optional color split), and prints Spearman correlation.

    Args:
        df: DataFrame
        x_col, y_col: column names for x and y axes
        x_label, y_label: optional axis labels
        alpha: point transparency
        xlim, ylim: optional limits for axes
        color_split_y: if True, colors points differently for y >= 0 vs y < 0
        above_color, below_color: colors for y >= 0 and y < 0 points
    """
    x_label = x_label or x_col
    y_label = y_label or y_col

    tmp = remove_outliers(df.copy(), [x_col, y_col])

    if color_split_y:
        above = tmp[tmp[y_col] >= 0]
        below = tmp[tmp[y_col] < 0]
        plt.scatter(above[x_col], above[y_col], alpha=alpha, color=above_color, label=f"{y_col} ≥ 0")
        plt.scatter(below[x_col], below[y_col], alpha=alpha, color=below_color, label=f"{y_col} < 0")
        plt.legend()
    else:
        plt.scatter(tmp[x_col], tmp[y_col], alpha=alpha)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.show()

    spearman_summary(tmp[x_col], tmp[y_col], x_label, y_label)

def select_best_slack_weight(df, metric="MAE"):
    results = []

    for model in df['model'].unique():
        for var in df['var'].unique():
            for iter in df['iter'].unique():
                df_subset = df[(df['var'] == var) & (df['iter'] == iter) & (df['model'] == model)]

                # find row that minimizes MAE
                try:
                    best_idx = df_subset[f'METimE_{metric}'].idxmin()
                    best_row = df_subset.loc[best_idx]
                    #print("Optimal slack weight: ", best_row['slack_weight'])
                    results.append(best_row)
                except:
                    print("error")



    # return a DataFrame of the selected best rows
    return pd.DataFrame(results).reset_index(drop=True)


def metrics_per_slack_weight(df):
    # Calculate group averages
    df_mean = df.groupby(["slack_weight", "var"])[
        ["METimE_AIC", "METimE_MAE", "METimE_RMSE", "METE_AIC", "METE_MAE", "METE_RMSE"]
    ].mean().reset_index()

    # Create figure with shared x-axis
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # # Colormap for consistent census colors
    # cmap = plt.get_cmap("tab10", df_mean["iter"].nunique())
    # census_colors = {c: cmap(i) for i, c in enumerate(sorted(df["iter"].unique()))}

    # --- Top subplot: MAE ---
    for var, group in df.groupby("var"):
        axes[0].plot(group["slack_weight"], group["METimE_MAE"], label=f"Variance {var}")
        #axes[0].plot(group["slack_weight"], group["METE_MAE"], color=color, linestyle="--")

    # Average lines
    # axes[0].plot(df_mean["slack_weight"], df_mean["METimE_MAE"],
    #              color="black", linewidth=2)
    # axes[0].plot(df_mean["slack_weight"], df_mean["METE_MAE"],
    #              color="black", linewidth=2, linestyle="--")

    axes[0].set_ylabel("MAE")
    axes[0].grid(False)
    #axes[0].legend(ncol=2)

    # --- Bottom subplot: RMSE ---
    for var, group in df.groupby("var"):
        axes[1].plot(group["slack_weight"], group["METimE_RMSE"], label=f"Variance {var}")
        #axes[1].plot(group["slack_weight"], group["METE_RMSE"], color=color, linestyle="--")

    # Average lines
    # axes[1].plot(df_mean["slack_weight"], df_mean["METimE_RMSE"],
    #              color="black", linewidth=2)
    # axes[1].plot(df_mean["slack_weight"], df_mean["METE_RMSE"],
    #              color="black", linewidth=2, linestyle="--")

    axes[1].set_xlabel("Slack weight")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(False)
    #axes[1].legend(ncol=2)

    plt.xscale('log')

    # --- Bottom subplot: AIC ---
    for var, group in df.groupby("var"):
        axes[2].plot(group["slack_weight"], group["METimE_AIC"], label=f"Variance {var}")
        #axes[2].plot(group["slack_weight"], group["METE_AIC"], color=color, linestyle="--")

    # Average lines
    # axes[2].plot(df_mean["slack_weight"], df_mean["METimE_AIC"],
    #              color="black", linewidth=2)
    # axes[2].plot(df_mean["slack_weight"], df_mean["METE_AIC"],
    #              color="black", linewidth=2, linestyle="--")

    axes[2].set_ylabel("AIC")
    axes[2].grid(False)
    #axes[2].legend(ncol=2)

    plt.tight_layout()
    plt.show()


def cleaner_look_single(df, ext):
    dark_greyish = "#4c4c4c"
    greyish = "#707070"

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    numeric_cols = [
        'METE_AIC', 'METimE_AIC', 'METE_MAE', 'METimE_MAE',
        'METE_RMSE', 'METimE_RMSE', 'METE_error_N/S', 'METimE_error_N/S',
        'METE_error_dN/S', 'METimE_error_dN/S'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute differences
    df_diff = pd.DataFrame({
        'model_code': df['model'],
        'var': df['var'],
        'iter': df['iter'],
        'AIC': (df['METE_AIC'] - df['METimE_AIC']) / df['METE_AIC'],
        'MAE': (df['METE_MAE'] - df['METimE_MAE']) / df['METE_MAE'],
        'RMSE': (df['METE_RMSE'] - df['METimE_RMSE']) / df['METE_RMSE'],
        'N/S error': (df['METE_error_N/S'] - df['METimE_error_N/S']) / df['METE_error_N/S']
    })

    # Create mapping for nicer model names
    model_labels = {
        "a": "(a) Constant",
        "b": "(b) Two predators-one prey",
        "c": "(c) One predator-two preys",
        "d": "(d) Food chain with cycle",
        "e": "(e) Food chain",
        "f": "(f) Food chain with omnivory"
    }

    # Apply the mapping
    df_diff['model'] = df_diff['model_code'].map(model_labels)
    df['model'] = df['model'].map(model_labels)

    # ➡️ Melt to long format
    df_long = df_diff.melt(
        id_vars=['model', 'var', 'iter'],
        value_vars=['MAE', 'RMSE'],
        var_name='Metric',
        value_name='Relative difference'
    )

    df_original = df_long.copy()

    for var in df_original['var'].unique():
        df_long = df_original[df_original['var'] == var]
        plt.figure(figsize=(8, 6))

        ax = sns.boxplot(
            x='Metric',
            y='Relative difference',
            hue='model',
            data=df_long,
            palette='Set2',
            showfliers=False,
            linewidth=1.5,
            showmeans=True,
            medianprops={
                "color": dark_greyish,
                "linewidth": 3
            },
            meanprops={
                "marker": "o",  # circle marker
                "markerfacecolor": greyish,
                "markeredgecolor": dark_greyish,
                "markersize": 6  # adjust size as needed
            }
        )

        sns.move_legend(ax, "lower center",
                        bbox_to_anchor=(.5, 1),
                        ncol=3,
                        title="Lotka-Volterra interaction network",
                        frameon=False)

        # Strong horizontal line at 0
        ax.axhline(0, color=greyish, linewidth=2, linestyle="--", zorder=1)

        ax.set_xlabel("")
        ax.set_ylabel("Relative difference \n (METE - METimE) / METE", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        #ax.set_title("Comparison of Metrics by Fraction Removed", fontsize=20)

        #ax.set_yscale("symlog")
        #ax.set_ylim(-45, 3)
        #plt.legend(title="Lotka-Volterra model", fontsize=12, title_fontsize=13)
        plt.tight_layout()
        plt.savefig(
            f"C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/LV_for_revisions{ext}/LV_boxplot_var={var}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ➡️ Prepare data for AIC violin plot
        df_aic = pd.melt(
            df[df['var'] == var],
            id_vars=['iter', 'model', 'var'],
            value_vars=['METE_AIC', 'METimE_AIC'],
            var_name='Model',
            value_name='AIC'
        )

        # Rename Models
        df_aic['Method'] = df_aic['Model'].replace({
            'METE_AIC': 'METE',
            'METimE_AIC': 'METimE'
        })

        plt.figure(figsize=(6, 8))

        g = sns.catplot(
            x="model",
            y="AIC",
            hue="Method",
            data=df_aic,
            kind="box",
            linewidth=1.5,
            showfliers=False
        )

        sns.move_legend(g, "upper left",
                        bbox_to_anchor=(0.25, 0.98),
                        title = "MaxEnt method",
                        frameon=True)

        # 🎨 Clean style
        g.set_xlabels("")
        g.set_ylabels("AIC", fontsize=18)
        g.tick_params(axis='both', which='major', labelsize=14)
        sns.despine()
        #ax.set_yscale("symlog")
        # g.ax.legend(loc="upper left")
        # g.legend.set_title("MaxEnt method")  # optional
        plt.xticks(rotation=30, ha="right")
        #plt.legend(handles, labels, title="MaxEnt method", fontsize=12, title_fontsize=13, loc="upper left")

        #plt.ylim(0, 550)

        plt.tight_layout()
        plt.savefig(
            f"C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/LV_for_revisions{ext}/LV_AIC_violin_var={var}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        #plt.show()


def scatterplot(df, ext):
    greyish = "#4c4c4c"
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Ensure numeric
    numeric_cols = [
        'METE_AIC', 'METimE_AIC', 'METE_MAE', 'METimE_MAE',
        'METE_RMSE', 'METimE_RMSE', 'METE_error_N/S', 'METimE_error_N/S',
        'METE_error_dN/S', 'METimE_error_dN/S'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    metrics = ['AIC', 'MAE', 'RMSE', 'error_N/S', 'error_dN/S']
    METE_metrics = ['METE_' + m for m in metrics]
    METimE_metrics = ['METimE_' + m for m in metrics]

    # # Create mapping for nicer model names
    # model_labels = {
    #     "a": "(a) Constant",
    #     "b": "(b) Two predators-one prey",
    #     "c": "(c) One predator-two preys",
    #     "d": "(d) Food chain with cycle",
    #     "e": "(e) Food chain",
    #     "f": "(f) Food chain with omnivory"
    # }
    #
    # # Apply the mapping
    # df['model'] = df['model'].map(model_labels)

    for var in df['var'].unique():
        df_var = df[df['var'] == var]

        # Create 2x3 grid (6 spots, but we only need 5)
        fig, axes = plt.subplots(2, 3, figsize=(12, 10), sharey=False)
        axes = axes.flatten()

        # We'll capture handles/labels from the FIRST plot for the global legend
        handles, labels = None, None

        for i, (metric, METE_metric, METimE_metric) in enumerate(zip(metrics, METE_metrics, METimE_metrics)):
            ax = axes[i]

            sns.scatterplot(
                x=METE_metric,
                y=METimE_metric,
                data=df_var,
                hue="model",
                palette="Set2",
                ax=ax,
                s=80,
                alpha=0.8,
                edgecolor=greyish,
                linewidth=1.0
            )

            # Capture legend info (only once)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()

            # Add x=y line
            lims = [
                np.nanmin([ax.get_xlim(), ax.get_ylim()]),
                np.nanmax([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, '--', color=greyish, lw=1.5, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")

            # Styling
            ax.set_xlabel("METE", fontsize=18)
            ax.set_ylabel("METimE", fontsize=18)

            if metric == "error_N/S":
                ax.set_title(r"$\frac{N}{S}$ violation", fontsize=20)
            elif metric == "error_dN/S":
                ax.set_title(r"$\frac{\Delta N}{S}$ violation", fontsize=20)
            else:
                ax.set_title(metric, fontsize=20)
            ax.tick_params(axis="both", which="major", labelsize=14)

        # Remove unused last subplot
        fig.delaxes(axes[-1])

        # Remove per-axes legends
        for ax in axes[:-1]:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Global legend underneath
        fig.legend(handles, labels, title="Lotka-Volterra \n interaction network", loc="upper center", ncol=1, frameon=False, title_fontsize='xx-large', fontsize='xx-large')
        sns.move_legend(fig,
                        "upper left",
                        bbox_to_anchor=(0.7, .45),
                        frameon=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.85)

        plt.savefig(
            f"C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/LV_for_revisions{ext}/LV_scatterplot_var={var}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        #plt.show()


def transition_functions(df, ext):
    dark_greyish = "#4c4c4c"
    greyish = "#707070"

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Ensure numeric
    df['r2_dn'] = pd.to_numeric(df['r2_dn'], errors='coerce')

    # # Create mapping for nicer model names
    # model_labels = {
    #     "a": "(a) Constant",
    #     "b": "(b) Two predators-one prey",
    #     "c": "(c) One predator-two preys",
    #     "d": "(d) Food chain with cycle",
    #     "e": "(e) Food chain",
    #     "f": "(f) Food chain with omnivory"
    # }
    #
    # # Apply the mapping
    # df['model'] = df['model'].map(model_labels)

    # Create two separate DataFrames
    df_dn = pd.DataFrame({
        'R^2': df['r2_dn'],
        'model': df['model'],  # use the pretty names
        'var': df['var']
    })

    ylim_min = df_dn['R^2'].min() - 0.02
    ylim_max = df_dn['R^2'].max() + 0.02

    for var in df['var'].unique():
        df_dn2 = df_dn[df_dn['var'] == var]

        plt.figure(figsize=(6, 8))

        sns.boxplot(
            x='model',
            y='R^2',
            data=df_dn2,
            palette='Set2',
            showfliers=True,
            linewidth=1.5,
            showmeans=True,
            medianprops={"color": dark_greyish, "linewidth": 3},
            meanprops={
                "marker": "o",
                "markerfacecolor": greyish,
                "markeredgecolor": dark_greyish,
                "markersize": 6
            }
        )

        plt.xlabel("", fontsize=18)
        #plt.xlabel("Lotka-Volterra interaction network", fontsize=18)
        plt.ylabel("Coefficient of determination (R²)", fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.xticks(rotation=30, ha="right")

        plt.ylim(ylim_min, ylim_max)

        plt.tight_layout()
        plt.savefig(
            f"C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/LV_for_revisions{ext}/LV_transition_functions_var={var}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        #plt.show()

def transition_functions_mae(df):
    dark_greyish = "#4c4c4c"
    greyish = "#707070"

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Ensure numeric
    df['mae_dn'] = pd.to_numeric(df['r2_dn'], errors='coerce')

    # # Create mapping for nicer model names
    # model_labels = {
    #     "a": "(a) Constant",
    #     "b": "(b) Two predators-one prey",
    #     "c": "(c) One predator-two preys",
    #     "d": "(d) Food chain with cycle",
    #     "e": "(e) Food chain",
    #     "f": "(f) Food chain with omnivory"
    # }
    #
    # # Apply the mapping
    # df['model_pretty'] = df['model'].map(model_labels)

    # Create two separate DataFrames
    df_dn = pd.DataFrame({
        'R^2': df['r2_dn'],
        'model': df['model'],  # use the pretty names
        'var': df['var']
    })

    for var in df['var'].unique():
        df_dn2 = df_dn[df_dn['var'] == var]

        plt.figure(figsize=(6, 8))

        sns.boxplot(
            x='model',
            y='R^2',
            data=df_dn2,
            palette='Set2',
            showfliers=True,
            linewidth=1.5,
            showmeans=True,
            medianprops={"color": dark_greyish, "linewidth": 3},
            meanprops={
                "marker": "o",
                "markerfacecolor": greyish,
                "markeredgecolor": dark_greyish,
                "markersize": 6
            }
        )

        plt.xlabel("Lotka-Volterra interaction network", fontsize=18)
        plt.ylabel("Coefficient of determination (R²)", fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        plt.savefig(
            f"LV_transition_functions_var={var}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()


def do_statistics(df, ext):
    results = []  # Store test results

    for model in df['model'].unique():
        for var in df['var'].unique():
            df_model = df[(df['model'] == model) & (df['var'] == var)]

            # Ensure we have both columns
            if not {'METE_MAE', 'METimE_MAE'}.issubset(df_model.columns):
                raise ValueError("DataFrame must contain 'METE_MAE' and 'METimE_MAE' columns.")

            # # Plot histogram
            # plt.figure(figsize=(6, 4))
            # plt.hist(diff, bins=10, color='steelblue', edgecolor='black')
            # plt.title(f"Difference Histogram: {model} - {var}")
            # plt.xlabel("METE_MAE - METimE_MAE")
            # plt.ylabel("Frequency")
            # plt.tight_layout()
            # plt.show()

            diff = df_model['METE_MAE'] - df_model['METimE_MAE']

            wilcoxon_res = wilcoxon(df_model['METE_MAE'], df_model['METimE_MAE'], method="asymptotic")
            p_val_wilcoxon = wilcoxon_res.pvalue
            z_val_wilcoxon = wilcoxon_res.zstatistic

            # Collect results
            results.append({
                'model': model,
                'var': var,
                'wilcoxon_p': p_val_wilcoxon,
                'wilcoxon_z': z_val_wilcoxon,
                'median_METE': np.median(df_model['METE_MAE']),
                'median_METimE': np.median(df_model['METimE_MAE']),
                'METE-METimE': np.median(df_model['METE_MAE']) - np.median(df_model['METimE_MAE'])
            })

    results_df = pd.DataFrame(results)

    results_df = results_df[['model', 'var', 'median_METE', 'median_METimE', 'METE-METimE', 'wilcoxon_p', 'wilcoxon_z']]

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

    path = f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/LV_for_revisions{ext}/statistics.csv'
    results_df.to_csv(path, index=False)

    return results_df

def summarize_results_latex(df: pd.DataFrame) -> str:
    """
    Compute summary stats grouped by 'frac' and return a LaTeX table.

    Expected columns in df:
        'frac', 'METE_MAE', 'METE_RMSE', 'METE_NS', 'METE_ES',
        'METimE_MAE', 'METimE_RMSE', 'METimE_NS', 'METimE_ES'
    """
    # 1️⃣ Select best slack weight per frac if needed
    df = select_best_slack_weight(df, 'MAE')

    metrics = ['MAE', 'RMSE', 'error_N/S']
    table_rows = []

    for model, df_model in df.groupby(['model']):
        # Outliers for METimE_MAE
        diff_ratio = (df_model['METE_MAE'] - df_model['METimE_MAE']) / df_model['METE_MAE']
        q1 = diff_ratio.quantile(0.25)
        iqr = diff_ratio.quantile(0.75) - q1
        lower_bound = q1 - 1.5 * iqr
        outliers = diff_ratio < lower_bound
        pct_outliers = 100 * outliers.mean()  # percentage of 20 iterations

        for m in metrics:
            mete = df_model[f"METE_{m}"]
            metime = df_model[f"METimE_{m}"]
            table_rows.append([
                model, m,
                f"{mete.max():.3f}",
                f"{metime.max():.3f}",
                f"{pct_outliers:.1f}" if m == 'MAE' else ""  # only for MAE
            ])

    # 2️⃣ Make LaTeX table
    header = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\begin{tabular}{l l c c c}\n"
        "\\toprule\n"
        "Interaction network & Metric & METE Max & METimE Max & METimE MAE Outliers (\\%) \\\\\n"
        "\\midrule\n"
    )

    body = "\n".join(
        f"{frac} & {m} & {mete_max} & {metime_max} & {outliers} \\\\"
        for frac, m, mete_max, metime_max, outliers in table_rows
    )

    footer = "\\bottomrule\n\\end{tabular}\n\\caption{Summary stats by frac, showing max values and METimE MAE outliers.}\n\\label{tab:mete_metime_by_frac}\n\\end{table}"

    return header + body + "\n" + footer


if __name__ == '__main__':

    for ext in ["_T=5", "_T=8", "_T=10"]:
        path = f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/LV_for_revisions{ext}'
        all_files = glob.glob(os.path.join(path, "*.csv"))

        dfs = []
        for f in all_files:
            if f.endswith("statistics.csv"):
                continue  # Skip this file

            # Extract variance level from filename
            match = re.search(r'results_([^_]+)\.csv$', f)
            model = match.group(1) if match else None

            temp_df = pd.read_csv(f)
            temp_df['model'] = model  # add as a new column
            dfs.append(temp_df)

        df = pd.concat(dfs, ignore_index=True)

        # for model in ['a', 'b', 'c', 'd', 'e', 'f']:
        #     df_model = df[df['model'] == model]
        #     metrics_per_slack_weight(df_model)

        df = select_best_slack_weight(df, 'MAE')

        cleaner_look_single(df, ext)

        transition_functions(df, ext)

        scatterplot(df, ext)

        do_statistics(df, ext)

