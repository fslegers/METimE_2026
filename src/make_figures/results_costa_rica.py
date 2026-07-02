import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ttest_rel, wilcoxon
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from matplotlib.font_manager import FontProperties


# --- Format numbers ---
def format_value(x, col=None):
    if isinstance(x, (int, float)):
        if col == "slack_weight":
            return f"{x:.1e}"  # scientific notation, 1 decimal
        else:
            return f"{x:.2f}".rstrip("0").rstrip(".")  # normal 2-decimal format
    return x

def metrics_per_slack_weight(df):
    # Calculate group averages
    df_mean = df.groupby("slack_weight")[
        ["METimE_AIC", "METimE_MAE", "METimE_RMSE", "METE_AIC", "METE_MAE", "METE_RMSE"]
    ].mean().reset_index()

    # Create figure with shared x-axis
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # Colormap for consistent census colors
    cmap = plt.get_cmap("tab10", df["census"].nunique())
    census_colors = {c: cmap(i) for i, c in enumerate(sorted(df["census"].unique()))}

    # --- Top subplot: MAE ---
    for census, group in df.groupby("census"):
        color = census_colors[census]
        axes[0].plot(group["slack_weight"], group["METimE_MAE"], color=color, label=f"Census {census}")
        axes[0].plot(group["slack_weight"], group["METE_MAE"], color=color, linestyle="--")

    # Average lines
    axes[0].plot(df_mean["slack_weight"], df_mean["METimE_MAE"],
                 color="black", linewidth=2)
    axes[0].plot(df_mean["slack_weight"], df_mean["METE_MAE"],
                 color="black", linewidth=2, linestyle="--")

    axes[0].set_ylabel("MAE")
    axes[0].grid(False)
    axes[0].legend(ncol=2)

    # --- Bottom subplot: RMSE ---
    for census, group in df.groupby("census"):
        color = census_colors[census]
        axes[1].plot(group["slack_weight"], group["METimE_RMSE"], color=color, label=f"Census {census}")
        axes[1].plot(group["slack_weight"], group["METE_RMSE"], color=color, linestyle="--")

    # Average lines
    axes[1].plot(df_mean["slack_weight"], df_mean["METimE_RMSE"],
                 color="black", linewidth=2)
    axes[1].plot(df_mean["slack_weight"], df_mean["METE_RMSE"],
                 color="black", linewidth=2, linestyle="--")

    axes[1].set_xlabel("Slack weight")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(False)
    axes[1].legend(ncol=2)

    plt.xscale('log')

    # --- Bottom subplot: AIC ---
    for census, group in df.groupby("census"):
        color = census_colors[census]
        axes[2].plot(group["slack_weight"], group["METimE_AIC"], color=color, label=f"Census {census}")
        axes[2].plot(group["slack_weight"], group["METE_AIC"], color=color, linestyle="--")

    # Average lines
    axes[2].plot(df_mean["slack_weight"], df_mean["METimE_AIC"],
                 color="black", linewidth=2)
    axes[2].plot(df_mean["slack_weight"], df_mean["METE_AIC"],
                 color="black", linewidth=2, linestyle="--")

    axes[2].set_ylabel("AIC")
    axes[2].grid(False)
    axes[2].legend(ncol=2)

    plt.tight_layout()
    plt.show()

# def fill_latex_table(df, plot):
#     # # --- Clean quad column ---
#     # df["quad"] = df["quad"].str.replace("_quadrat_", "", regex=False)
#
#     # Apply formatting column by column
#     for col in df.columns:
#         df[col] = df[col].apply(lambda x: format_value(x, col))
#
#     # --- Reorder columns (added slack_weight after quad and census) ---
#     cols = [
#         "PlotName", "census", "r2_dn", "r2_de", "METE_AIC", "METE_MAE", "METE_RMSE",
#         "METimE_AIC", "METimE_MAE", "METimE_RMSE"
#     ]
#     df = df[cols]
#
#     # --- Convert to LaTeX without headers ---
#     latex_table = df.to_latex(
#         index=False,
#         header=False,
#         column_format="cc|cc|ccc|ccc",
#         escape=False
#     )
#
#     # --- Build custom header (added slack_weight) ---
#     custom_header = (
#         "\\toprule\n"
#         " & & & &  \\multicolumn{3}{c|}{METE} & \\multicolumn{3}{c}{METimE} \\\\\n"
#         "PlotName & Census & r2_dn & r2_de & AIC & MAE & RMSE & AIC & MAE & RMSE \\\\\n"
#         "\\midrule\n"
#     )
#
#     # --- Insert header ---
#     latex_table = latex_table.replace("\\toprule", custom_header, 1)
#
#     # --- Save to file ---
#     with open(f"costa_rica_{plot}.tex", "w") as f:
#         f.write(latex_table)
#
#     print(latex_table)

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

# def print_additional_metrics(df):
#     n = len(df)
#
#     # Calculate how often METimE outperforms METE
#     better_AIC   = (df["METE_AIC"]   > df["METimE_AIC"]).sum()   / n * 100
#     better_MAE   = (df["METE_MAE"]   > df["METimE_MAE"]).sum()   / n * 100
#     better_RMSE  = (df["METE_RMSE"]  > df["METimE_RMSE"]).sum()  / n * 100
#     better_NS    = (df["METE_error_N/S"]  > df["METimE_error_N/S"]).sum() / n * 100
#     better_ES    = (df["METE_error_E/S"]  > df["METimE_error_E/S"]).sum() / n * 100
#     better_NoverS= (df["METE_error_dN/S"] > df["METimE_error_dN/S"]).sum() / n * 100
#     better_EoverS= (df["METE_error_dE/S"] > df["METimE_error_dE/S"]).sum() / n * 100
#
#     # Calculate how often they're equal
#     equal_AIC   = (df["METE_AIC"]   == df["METimE_AIC"]).sum()   / n * 100
#     equal_MAE   = (df["METE_MAE"]   == df["METimE_MAE"]).sum()   / n * 100
#     equal_RMSE  = (df["METE_RMSE"]  == df["METimE_RMSE"]).sum()  / n * 100
#     equal_NS    = (df["METE_error_N/S"]  == df["METimE_error_N/S"]).sum() / n * 100
#     equal_ES    = (df["METE_error_E/S"]  == df["METimE_error_E/S"]).sum() / n * 100
#     equal_NoverS= (df["METE_error_dN/S"] == df["METimE_error_dN/S"]).sum() / n * 100
#     equal_EoverS= (df["METE_error_dE/S"] == df["METimE_error_dE/S"]).sum() / n * 100
#
#     # Build summary table with LaTeX-friendly labels
#     summary = pd.DataFrame({
#         "Metric": [
#             "AIC", "MAE", "RMSE", r"$N/S$ error", r"$E/S$ error",
#             r"$\Delta N/S$ error", r"$\Delta E/S$ error"
#         ],
#         "METimE better than METE (\\%)": [
#             better_AIC, better_MAE, better_RMSE, better_NS, better_ES, better_NoverS, better_EoverS
#         ],
#         "METE as good as METimE (\\%)": [
#             equal_AIC, equal_MAE, equal_RMSE, equal_NS, equal_ES, equal_NoverS, equal_EoverS
#         ]
#     })
#
#     # Format as LaTeX table
#     latex_table = summary.to_latex(
#         index=False,
#         escape=False,  # keep LaTeX math symbols
#         float_format="%.2f"
#     )
#
#     print(latex_table)

# def how_much_difference(df):
#     #sns.set_theme(style="white")
#     #custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False}
#     #sns.set_theme(style="ticks", rc=custom_params)
#
#     # Colors
#     blueish = "#67a9cf"
#     greyish = "#4c4c4c"
#     orangy = "#ef8a62"
#
#     # Ensure numeric
#     for col in ['METE_AIC', 'METimE_AIC', 'METE_MAE', 'METimE_MAE', 'METE_RMSE', 'METimE_RMSE', 'METE_error_N/S', 'METimE_error_N/S', 'METE_error_E/S', 'METimE_error_E/S', 'METE_error_dN/S', 'METimE_error_dN/S', 'METE_error_dE/S', 'METimE_error_dE/S']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#
#     # Compute differences
#     df_diff = pd.DataFrame({
#         'AIC': df['METE_AIC'] - df['METimE_AIC'],
#         'MAE': df['METE_MAE'] - df['METimE_MAE'],
#         'RMSE': df['METE_RMSE'] - df['METimE_RMSE'],
#         'N/S error': df['METE_error_N/S'] - df['METimE_error_N/S'],
#         'E/S error': df['METE_error_E/S'] - df['METimE_error_E/S'],
#         'dN/S error': df['METE_error_dN/S'] - df['METimE_error_dN/S'],
#         'dE/S error': df['METE_error_dE/S'] - df['METimE_error_dE/S']
#     })
#
#     metrics = ['AIC', 'MAE', 'RMSE', 'N/S error', 'E/S error', 'dN/S error', 'dE/S error']
#
#     fig, axes = plt.subplots(1, 7, figsize=(24, 5), sharey=False)
#
#     for i, (ax, metric) in enumerate(zip(axes, metrics)):
#         values = df_diff[metric].dropna()
#
#         # Violin plot without inner box
#         sns.violinplot(
#             y=values, ax=ax, color=blueish, alpha=0.8,
#             inner=None, bw_adjust=0.5, cut=0, zorder=1
#         )
#
#         # Overlay custom boxplot (smaller width, rounded, black fill, white median)
#         sns.boxplot(
#             y=values, ax=ax, width=0.1, showcaps=False, showfliers=True,
#             boxprops=dict(facecolor=greyish, edgecolor=greyish, linewidth=1.2),
#             whiskerprops=dict(color=greyish, linewidth=1.0),
#             capprops=dict(color=greyish, linewidth=1.0),
#             medianprops=dict(color="white", linewidth=3),
#             flierprops=dict(markerfacecolor=greyish, markersize=5, alpha=0.5)
#         )
#
#         # Strong horizontal line at 0
#         ax.axhline(0, color=orangy, linewidth=4, linestyle="-", zorder=0)
#
#         # Get y-limits after plotting
#         ylim_min, ylim_max = ax.get_ylim()
#         #ax.axhspan(0, ylim_max, facecolor='lightgreen', alpha=0.3, zorder=0)
#
#         # Compute percentage above 0
#         total = len(values)
#         above = (values > 0).sum()
#         perc_above = above / total * 100 if total > 0 else 0
#
#         # Annotate percentages
#         ax.text(0.5, 0.98, f"{perc_above:.1f}% > 0",
#                 ha='center', va='top', transform=ax.transAxes,
#                 fontsize=14)
#
#         ax.set_title(metric, fontsize=16)
#
#         if i == 0:
#             ax.set_ylabel("Difference (METE - METimE)", fontsize=14)
#         else:
#             ax.set_ylabel("")
#
#         ax.tick_params(axis='both', which='major', labelsize=12)
#
#         # Inset boxplot on the right, centered vertically
#         inset_ax = inset_axes(ax, width="25%", height="40%", loc="lower right",
#                               borderpad=1.2)
#
#         sns.boxplot(
#             y=values, ax=inset_ax, width=0.2, showcaps=True, showfliers=False,
#             boxprops=dict(facecolor=greyish, edgecolor="black", linewidth=1.2),
#             whiskerprops=dict(color=greyish, linewidth=1.0),
#             capprops=dict(color=greyish, linewidth=1.0),
#             medianprops=dict(color="white", linewidth=3.0)
#         )
#
#         inset_ax.axhline(0, color=orangy, linewidth=4, linestyle="-")
#         inset_ax.set_xticks([])
#         inset_ax.set_xlabel("")
#         inset_ax.set_ylabel("")
#         inset_ax.tick_params(axis='y', labelsize=8)
#
#         ylim_min, ylim_max = ax.get_ylim()
#         ylim_max *= 1.80  # scale max by 10%
#         ax.set_ylim(ylim_min, ylim_max)
#
#     plt.tight_layout()
#     plt.show()

# def simple_violin(df):
#     """
#     Make a single figure with horizontal subplots:
#     - Each subplot shows half-violin comparisons (METE vs METimE) for one metric.
#     - Each subplot keeps its own y-axis.
#     """
#
#     # Colors
#     blueish = "#67a9cf"
#     orangy = "#ef8a62"
#
#     custom_params = {"axes.spines.right": False, "axes.spines.top": False}
#     sns.set_theme(style="ticks", rc=custom_params)
#
#     metrics = [
#         'AIC', 'MAE', 'RMSE', 'error_N/S', 'error_E/S',
#         'error_dN/S', 'error_dE/S'
#     ]
#
#     # Filter to metrics present in the DataFrame
#     valid_metrics = []
#     for m in metrics:
#         if f"METE_{m}" in df.columns and f"METimE_{m}" in df.columns:
#             valid_metrics.append(m)
#     n_metrics = len(valid_metrics)
#     if n_metrics == 0:
#         print("No valid metrics found.")
#         return
#
#     # Create a wide figure with one column per metric
#     fig, axes = plt.subplots(
#         1, n_metrics,
#         figsize=(4 * n_metrics, 6),
#         sharey=False  # independent y-axis for each metric
#     )
#     # axes is an array even if n_metrics == 1
#     if n_metrics == 1:
#         axes = [axes]
#
#     for ax, m in zip(axes, valid_metrics):
#         mete_col = f"METE_{m}"
#         metime_col = f"METimE_{m}"
#
#         # Tidy dataframe for this metric
#         plot_df = pd.concat([
#             pd.DataFrame({'Value': pd.to_numeric(df[mete_col], errors='coerce'),
#                           'Model': 'METE',
#                           'Metric': m}),
#             pd.DataFrame({'Value': pd.to_numeric(df[metime_col], errors='coerce'),
#                           'Model': 'METimE',
#                           'Metric': m})
#         ], ignore_index=True)
#
#         sns.violinplot(
#             data=plot_df,
#             x="Metric",
#             y="Value",
#             split=True,
#             inner="quart",
#             hue="Model",
#             palette={"METE": blueish, "METimE": orangy},
#             ax=ax,
#             density_norm="area",
#             inner_kws=dict(linewidth=2.5)
#         )
#
#         ax.set_title(m, fontsize=14)
#         ax.set_xlabel("")
#         ax.set_ylabel("Value", fontsize=12)
#         ax.tick_params(axis="both", labelsize=10)
#
#     plt.tight_layout()
#     plt.show()
#
# def cleaner_look(df):
#     greyish = "#4c4c4c"
#
#     custom_params = {"axes.spines.right": False, "axes.spines.top": False}
#     sns.set_theme(style="ticks", rc=custom_params)
#
#     # Ensure numeric
#     numeric_cols = [
#         'METE_AIC', 'METimE_AIC', 'METE_MAE', 'METimE_MAE',
#         'METE_RMSE', 'METimE_RMSE', 'METE_error_N/S', 'METimE_error_N/S',
#         'METE_error_E/S', 'METimE_error_E/S', 'METE_error_dN/S', 'METimE_error_dN/S',
#         'METE_error_dE/S', 'METimE_error_dE/S'
#     ]
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#
#     # Compute differences
#     df_diff = pd.DataFrame({
#         'frac': df['frac'],
#         'iter': df['iter'],
#         'AIC': (df['METE_AIC'] - df['METimE_AIC']) / df['METE_AIC'],
#         'MAE': (df['METE_MAE'] - df['METimE_MAE']) / df['METE_MAE'],
#         'RMSE': (df['METE_RMSE'] - df['METimE_RMSE']) / df['METE_RMSE'],
#         'N/S error': (df['METE_error_N/S'] - df['METimE_error_N/S']) / df['METE_error_N/S'],
#         'E/S error': (df['METE_error_E/S'] - df['METimE_error_E/S']) / df['METE_error_E/S']
#     })
#
#     metrics = ['AIC', 'MAE', 'RMSE', 'N/S error', 'E/S error']
#
#     fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=False)
#
#     for i, (ax, metric) in enumerate(zip(axes, metrics)):
#
#         # Use boxplot (handles repetitions via 'iter')
#         sns.boxplot(
#             x='frac',
#             y=metric,
#             data=df_diff,
#             palette='Set2',
#             hue='frac',
#             legend=False,
#             ax=ax,
#             showfliers=False,
#             linewidth=2
#         )
#
#         # Strong horizontal line at 0
#         ax.axhline(0, color=greyish, linewidth=3, linestyle="-", zorder=1)
#
#         if i == 0:
#             ax.set_ylabel("Relative difference", fontsize=18, linespacing=1.5)
#         else:
#             ax.set_ylabel("")
#
#         ax.set_xlabel("")
#         ax.tick_params(axis='both', which='major', labelsize=14)
#         ax.set_title(metric, fontsize=18)
#
#     # Figure-wide x-label
#     fig.text(0.5, 0.05, "Fraction of population removed", ha='center', fontsize=18)
#     fig.text(0.5, 0.95, "Metric", ha='center', fontsize=22)
#     plt.tight_layout(rect=[0, 0.1, 1, 0.9])
#     plt.show()

def cleaner_look_single(df, plotName):
    dark_greyish = "#4c4c4c"
    greyish = "#707070"
    blueish = "#67a9cf"
    orangy = "#ef8a62"

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    numeric_cols = [
        'METE_AIC', 'METimE_AIC', 'METE_MAE', 'METimE_MAE',
        'METE_RMSE', 'METimE_RMSE', 'METE_error_N/S', 'METimE_error_N/S',
        'METE_error_E/S', 'METimE_error_E/S', 'METE_error_dN/S', 'METimE_error_dN/S',
        'METE_error_dE/S', 'METimE_error_dE/S'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute differences
    df_diff = pd.DataFrame({
        'PlotName': df['PlotName'],
        'census': df['census'],
        'AIC': (df['METE_AIC'] - df['METimE_AIC']) / df['METE_AIC'],
        'MAE': (df['METE_MAE'] - df['METimE_MAE']) / df['METE_MAE'],
        'RMSE': (df['METE_RMSE'] - df['METimE_RMSE']) / df['METE_RMSE'],
        'N/S error': (df['METE_error_N/S'] - df['METimE_error_N/S']) / df['METE_error_N/S'],
        'E/S error': (df['METE_error_E/S'] - df['METimE_error_E/S']) / df['METE_error_E/S'],
        'dN/S error': (df['METE_error_dN/S'] - df['METimE_error_dN/S']) / df['METE_error_dN/S'],
        'dE/S error': (df['METE_error_dE/S'] - df['METimE_error_dE/S']) / df['METE_error_dE/S']
    })

    # ➡️ Melt to long format
    df_long = df_diff.melt(
        id_vars=['PlotName', 'census'],
        value_vars=['MAE', 'RMSE', 'N/S error', 'E/S error'],
        var_name='Metric',
        value_name='Relative difference'
    )

    # ✅ One big boxplot
    plt.figure(figsize=(4.5, 6))
    ax = sns.boxplot(
        x='Metric',
        y='Relative difference',
        color=blueish,
        data=df_long,
        showfliers=True,
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

    # Strong horizontal line at 0
    ax.axhline(0, color=greyish, linewidth=2, linestyle="--", zorder=1)
    plt.xticks(rotation=30, ha="right")

    ax.set_xlabel("")
    ax.set_ylabel("Relative difference \n (METE - METimE) / METE", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()

    # plt.savefig(
    #     "empirical_BCI_boxplot.png",
    #     dpi=300,
    #     bbox_inches="tight",
    #     transparent=True
    # )
    plt.title(plotName)
    plt.show()

    # ➡️ Prepare data for AIC violin plot
    df_aic = pd.melt(
        df,
        id_vars=['PlotName', 'census'],
        value_vars=['METE_AIC', 'METimE_AIC'],
        var_name='Model',
        value_name='AIC'
    )

    # ✅ Rename for cleaner legend/labels
    df_aic['Method'] = df_aic['Model'].replace({
        'METE_AIC': 'METE',
        'METimE_AIC': 'METimE'
    })

    # ✅ One big boxplot
    plt.figure(figsize=(4.5, 6))

    ax = sns.boxplot(
        x='Method',
        y='AIC',
        hue='Method',
        data=df_aic,
        showfliers=True,
        palette=[blueish, orangy],
        linewidth=1.5,
        showmeans=False,
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

    ax.set_xlabel("MaxEnt method", fontsize=18)
    ax.set_ylabel("AIC", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Save figure
    # plt.savefig(
    #     "costa_rica_{plot}_AIC_boxplot.png",
    #     dpi=300,
    #     bbox_inches="tight",
    #     transparent=True
    # )
    plt.title(plotName)
    plt.tight_layout()
    plt.show()

def scatterplot(df, plotName):
    greyish = "#4c4c4c"
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Ensure numeric
    numeric_cols = [
        'METE_AIC', 'METimE_AIC', 'METE_MAE', 'METimE_MAE',
        'METE_RMSE', 'METimE_RMSE', 'METE_error_N/S', 'METimE_error_N/S',
        'METE_error_E/S', 'METimE_error_E/S', 'METE_error_dN/S', 'METimE_error_dN/S',
        'METE_error_dE/S', 'METimE_error_dE/S'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    metrics = ['AIC', 'MAE', 'RMSE', 'error_N/S', 'error_E/S', 'error_dN/S', 'error_dE/S']
    METE_metrics = ['METE_' + m for m in metrics]
    METimE_metrics = ['METimE_' + m for m in metrics]

    fig, axes = plt.subplots(1, 7, figsize=(35, 5), sharey=False)

    # We'll capture handles/labels from the first subplot that actually produces legend entries
    handles, labels = None, None

    for ax, metric, METE_metric, METimE_metric in zip(axes, metrics, METE_metrics, METimE_metrics):

        # Scatter plot without legend
        sns.scatterplot(
            x=METE_metric,
            y=METimE_metric,
            data=df,
            hue="PlotName",
            palette="Set3",
            ax=ax,
            s=100,
            alpha=0.8,
            edgecolor=greyish,
            linewidth=1.0,
            legend=False
        )

        # Capture legend info if not already captured
        if handles is None:
            h, l = ax.get_legend_handles_labels()
            if h:  # only assign if handles exist
                handles, labels = h, l

        # Diagonal x=y line
        lims = [
            np.nanmin([ax.get_xlim(), ax.get_ylim()]),
            np.nanmax([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, '--', color=greyish, lw=1.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Styling
        ax.set_xlabel("METE", fontsize=12)
        ax.set_ylabel("METimE", fontsize=12)
        ax.set_title(metric, fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=10)

    # Add global legend at bottom if handles exist
    if handles:
        fig.legend(
            handles, labels, title=f"{plotName}", title_fontsize=13,
            loc="lower center", ncol=len(labels), frameon=False, fontsize=12
        )
    plt.suptitle(plotName)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])  # leave space for bottom legend
    # plt.savefig(
    #     "costa_rica_{plot}_scatterplot.png",
    #     dpi=300,
    #     bbox_inches="tight",
    #     transparent=True
    # )
    plt.show()

def summarize_results_latex(df: pd.DataFrame) -> str:
    """
    Compute summary stats and return a LaTeX table of results.

    Expected columns in df:
        'METE_MAE', 'METE_RMSE', 'METE_NS', 'METE_ES',
        'METimE_MAE', 'METimE_RMSE', 'METimE_NS', 'METimE_ES'
    """
    # 1️⃣ Select best slack weight
    df = select_best_slack_weight(df, 'MAE')

    # 2️⃣ Summary statistics
    metrics = ['MAE', 'RMSE', 'error_N/S', 'error_E/S']
    rows = []
    for m in metrics:
        mete = df[f"METE_{m}"]
        metime = df[f"METimE_{m}"]
        rows.append([
            m,
            f"{mete.min():.3f}", f"{mete.max():.3f}",
            f"{metime.min():.3f}", f"{metime.max():.3f}"
        ])

    # 3️⃣ Outliers
    diff_ratio = (df['METE_MAE'] - df['METimE_MAE']) / df['METE_MAE']
    q1 = diff_ratio.quantile(0.25)
    iqr = diff_ratio.quantile(0.75) - q1
    lower_bound = q1 - 1.5 * iqr
    outliers = diff_ratio < lower_bound
    pct_outliers = 100 * outliers.mean()

    # 4️⃣ Percentage of times METE_MAE > METimE_MAE (excluding outliers)
    mask = ~outliers
    pct_mete_worse = 100 * (df.loc[mask, 'METE_MAE'] > df.loc[mask, 'METimE_MAE']).mean()

    # 5️⃣ Make LaTeX table
    header = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Metric & METE Min & METE Max & METimE Min & METimE Max \\\\\n"
        "\\midrule\n"
    )

    body = "\n".join(
        f"{m} & {mete_min} & {mete_max} & {metime_min} & {metime_max} \\\\"
        for m, mete_min, mete_max, metime_min, metime_max in rows
    )

    footer = (
        "\\midrule\n"
        f"\\multicolumn{{5}}{{l}}{{Outliers: {pct_outliers:.1f}\\%}} \\\\\n"
        f"\\multicolumn{{5}}{{l}}{{METE worse than METimE (excl. outliers): {pct_mete_worse:.1f}\\%}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Summary statistics comparing METE and METimE.}\n"
        "\\label{tab:mete_metime_summary}\n"
        "\\end{table}"
    )

    return header + body + "\n" + footer

def transition_functions_boxplot(df, plotName):
    dark_greyish = "#4c4c4c"
    greyish = "#707070"
    blueish = "#67a9cf"
    orangy = "#ef8a62"

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Ensure numeric
    df['r2_dn'] = pd.to_numeric(df['r2_dn'], errors='coerce')
    df['r2_de'] = pd.to_numeric(df['r2_de'], errors='coerce')

    # Create separate DataFrames with correct "Transition function" labels
    df_dn = pd.DataFrame({
        'Transition function': 'f',
        'R^2': df['r2_dn'],
        'Metric': 'r2_dn'
    })
    df_de = pd.DataFrame({
        'Transition function': 'h',
        'R^2': df['r2_de'],
        'Metric': 'r2_de'
    })

    # Combine
    df_plot = pd.concat([df_dn, df_de], ignore_index=True)

    # Boxplot
    plt.figure(figsize=(4.5, 6))
    ax = sns.boxplot(
        x='Transition function',
        y='R^2',
        data=df_plot,
        showfliers=True,
        linewidth=1.5,
        showmeans=True,
        color=blueish,
        medianprops={"color": dark_greyish,
                     "linewidth": 3},
        meanprops={
            "marker": "o",
            "markerfacecolor": dark_greyish,
            "markeredgecolor": dark_greyish,
            "markersize": 6
        }
    )

    # # Horizontal line at 0
    # ax.axhline(0, color=greyish, linewidth=2, linestyle="--", zorder=1)

    ax.set_xlabel("")
    ax.set_xticklabels([
        r"$f \approx \Delta n$",
        r"$h \approx \Delta \overline{\varepsilon}$"], fontsize=18)

    ax.set_ylabel("Coefficient of determination (R²)", fontsize=18)
    ax.tick_params(axis='y', which='major', labelsize=14)  # y-axis smaller
    ax.tick_params(axis='x', which='major', labelsize=18)  # x-axis larger

    plt.title(plotName)
    plt.tight_layout()
    # plt.savefig(
    #     "costa_rica_{plot}_transition_functions.png",
    #     dpi=300,
    #     bbox_inches="tight",
    #     transparent=True
    # )
    plt.show()

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

def plot_time_series(df_plot, plotName):
    # Get unique censuses (years)
    censuses = df_plot['census'].unique()
    censuses_sorted = sorted(censuses)  # Sort years for plotting

    # Create a figure with two vertically stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
    fig.suptitle(plotName)

    # Plot N/S in the first subplot
    ax1.plot(censuses_sorted, [df_plot[df_plot['census'] == year]['N/S'].values[0] for year in censuses_sorted],
                 marker='o', label='N/S', color='blue')
    ax1.set_ylabel('N/S')
    ax1.legend()

    # Plot E/S in the second subplot
    ax2.plot(censuses_sorted, [df_plot[df_plot['census'] == year]['E/S'].values[0] for year in censuses_sorted],
                 marker='o', label='E/S', color='red')
    ax2.set_ylabel('E/S')
    ax2.set_xlabel('Census (Year)')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def make_transition_function_table():
    plot_names = ['BEJ', 'CR', 'JE', 'LEP', 'LSUR', 'SV', 'TIR']

    table = []

    for i, plot in enumerate(plot_names):
        path = f'../MaxEnt_inference/costa_rica_2_df{plot}.csv'
        df_plot = pd.read_csv(path)
        r2_dn = round(df_plot['r2_dn'].unique()[0], 3)
        r2_de = round(df_plot['r2_de'].unique()[0], 3)
        table.append([plot, r2_dn, r2_de])

    return pd.DataFrame(table)


# def make_scatter_plot(metric):
#     sns.set_theme(context="poster", style="white")
#     plt.figure(figsize=(10, 8))
#
#     palette = sns.color_palette("husl", n_colors=8)
#     plot_names = ['BEJ', 'CR', 'JE', 'LEP', 'LEPviejo', 'LSUR', 'SV', 'TIR']
#     #plot_names = ['JE', 'CR', 'LEP', 'LSUR', 'SV', 'TIR']
#     handles = []
#     labels = []
#
#     for i, plot in enumerate(plot_names):
#         path = f'../MaxEnt_inference/results/costa_rica_df{plot}.csv'
#         #path=f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/PythonProjects/METimE_2026/METimE_2026/src/MaxEnt_inference/costa_rica_2_df{plot}.csv'
#         df_plot = pd.read_csv(path)
#         df = select_best_slack_weight(df_plot, metric)
#
#         x_median = np.median(df[f"METE_{metric}"])
#         y_median = np.median(df[f"METimE_{metric}"])
#
#         sns.scatterplot(
#             data=df,
#             x=f"METE_{metric}",
#             y=f"METimE_{metric}",
#             color=palette[i],
#             label=plot,
#             alpha=0.5,
#             zorder=1
#         )
#
#         plt.scatter(x_median, y_median, color=palette[i], s=300, marker='X', edgecolors='black', linewidth=2, zorder=2)
#         #plt.text(x_median, y_median, plot, fontsize=12, ha='center', va='bottom', color=palette[i])
#
#         # Collect handles and labels for custom legend
#         handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10, markeredgecolor='none'))
#         labels.append(plot)
#
#     # Add 1-1 line
#     max_val = max(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1])
#     plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1, zorder=0)
#
#     # Force square aspect
#     plt.gca().set_aspect('equal')
#
#     plt.xlabel(f"METE")
#     plt.ylabel(f"METimE")
#     plt.title(f"{metric}")
#
#     # Split legend into three groups
#     legend1 = plt.legend(
#         handles=[handles[plot_names.index(p)] for p in ['LEPviejo', 'SV']],
#         labels=['LEPviejo', 'SV'],
#         loc='upper left',
#         bbox_to_anchor=(1.05, 0.9),
#         frameon=False,
#         handletextpad=0.5,
#         markerfirst=True
#     )
#     plt.gca().add_artist(legend1)
#
#     legend2 = plt.legend(
#         handles=[handles[plot_names.index(p)] for p in ['TIR', 'CR', 'BEJ']],
#         labels=['TIR', 'CR', 'BEJ'],
#         loc='upper left',
#         bbox_to_anchor=(1.05, 0.7),
#         frameon=False,
#         handletextpad=0.5,
#         markerfirst=True
#     )
#     plt.gca().add_artist(legend2)
#
#     legend3 = plt.legend(
#         handles=[handles[plot_names.index(p)] for p in ['LSUR', 'LEP', 'JE']],
#         labels=['LSUR', 'LEP', 'JE'],
#         loc='upper left',
#         bbox_to_anchor=(1.05, 0.4),
#         frameon=False,
#         handletextpad=0.5,
#         markerfirst=True
#     )
#     plt.gca().add_artist(legend3)
#
#     plt.tight_layout()
#     plt.show()

def make_scatter_plot(metric):
    sns.set_theme(context="poster", style="white")
    plt.figure(figsize=(10, 8))

    palette = sns.color_palette("husl", n_colors=8)
    plot_names = ['BEJ', 'CR', 'JE', 'LEP', 'LEPviejo', 'LSUR', 'SV', 'TIR']

    for i, plot in enumerate(plot_names):

        if plot == "LEPviejo":
            continue

        path = f'../MaxEnt_inference/costa_rica_2_df{plot}.csv'
        df_plot = pd.read_csv(path)
        df = select_best_slack_weight(df_plot, metric)

        x_median = np.median(df[f"METE_{metric}"])
        y_median = np.median(df[f"METimE_{metric}"])

        sns.scatterplot(
            data=df,
            x=f"METE_{metric}",
            y=f"METimE_{metric}",
            color=palette[i],
            alpha=0.5,
            zorder=1
        )

        # Plot median cross
        plt.scatter(x_median, y_median, color=palette[i], s=300, marker='X', edgecolors='black', linewidth=2, zorder=2)

        # Add label next to the median cross
        # 'BEJ', 'CR', 'JE', 'LEP', 'LEPviejo', 'LSUR', 'SV', 'TIR'
        x_offset = [0, -0.9, 0, 0.9, 0, 1.8, 0, 0]
        y_offset = [0.9, 1.2, 0.9, -0.9, 0, 0, 0.9, 0.9]
        plt.text(
            x_median + x_offset[i],
            y_median + y_offset[i],
            plot,
            fontsize=16,
            ha = 'center',
            va = 'center',
            color='black',
            zorder=3
        )

        if i == 1:
            plt.plot(
                [x_median, x_median - 0.5],
                [y_median, y_median + 0.9],
                color='black',
                linewidth=2,
                alpha=1,
                zorder=2
            )

        if i == 3:
            plt.plot(
                [x_median, x_median + 0.5],
                [y_median, y_median + (y_offset[i] * 0.6)],
                color='black',
                linewidth=2.5,
                alpha=0.8,
                zorder=1
            )

    # Add 1-1 line
    max_val = max(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1])
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1, zorder=0)

    # Force square aspect
    plt.gca().set_aspect('equal')

    plt.xlabel(f"METE")
    plt.ylabel(f"METimE")
    plt.title(f"{metric}")

    plt.tight_layout()
    plt.show()
    #plt.savefig(f"../MaxEnt_inference/results/costa_rica_scatter_2_{metric}.png", dpi=300, bbox_inches="tight", transparent=True)

def scatter_per_year(df, metric, plot):
        sns.set_theme(context="poster", style="whitegrid")
        plt.figure(figsize=(10, 8))

        years = sorted(df['census'].unique())
        min_year, max_year = min(years), max(years)

        # Normalize years for colormap
        norm = plt.Normalize(min_year, max_year)
        cmap = sns.color_palette("rocket", as_cmap=True)

        # Scatter plot with color mapped to year
        scatter = plt.scatter(
            data=df,
            x=f"METE_{metric}",
            y=f"METimE_{metric}",
            c=df['census'],
            cmap=cmap,
            norm=norm,
            zorder=1
        )

        # Add 1-1 line
        max_val = max(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1])
        plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1, zorder=0)

        # Force square aspect
        plt.gca().set_aspect('equal')

        # Add colorbar with min/max years
        cbar = plt.colorbar(scatter, ax=plt.gca(), pad=0.02)
        cbar.set_ticks([min_year, max_year])
        cbar.set_ticklabels([str(min_year), str(max_year)])
        cbar.set_label('Year')

        plt.xlabel(f"METE")
        plt.ylabel(f"METimE")
        plt.title(plot)
        plt.tight_layout()
        plt.show()

        # Time series plots of metrics
        plt.figure(figsize=(10, 5))

        plt.scatter(years, [df[df['census'] == year][f'METE_{metric}'] for year in years])
        plt.plot(years, [df[df['census']==year][f'METE_{metric}'] for year in years], label="METE")

        plt.scatter(years, [df[df['census'] == year][f'METimE_{metric}'] for year in years])
        plt.plot(years, [df[df['census']==year][f'METimE_{metric}'] for year in years], label="METimE")

        plt.xlabel("Year")
        plt.ylabel(metric)
        plt.legend(loc="upper left", fontsize=18)
        plt.title(plot)

        # Add grid and custom ticks
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(ticks=range(int(min(years)), int(max(years)) + 1, 2))  # Ticks every 2 years
        plt.yticks(ticks=plt.yticks()[0])  # Keep default y-ticks
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Load data
    statistics_df = []
    tf_df = []
    metric = "MAE"

    for plot in ['BEJ', 'CR', 'JE', 'LEP', 'LSUR', 'SV', 'TIR']:
    #for plot in ['JE', 'CR', 'LEP', 'LSUR', 'SV', 'TIR']:
        path = f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/PythonProjects/METimE_2026/METimE_2026/src/MaxEnt_inference/costa_rica_2_df{plot}.csv'

        # all_files = glob.glob(os.path.join(path, "*.csv"))
        df_plot = pd.read_csv(path)

        #metrics_per_slack_weight(df_plot)

        #plot_time_series(df_plot, plot)

        df = select_best_slack_weight(df_plot, metric)

        # Also report how much better METimE predicts than METE on average, or vise versa, also reporting outliers
        #cleaner_look_single(df, plot)
        scatterplot(df, plot)
        #transition_functions_boxplot(df, plot)
        #latex_code = summarize_results_latex(df)
        #print(latex_code)

        # diff = (df['METimE_MAE'] - df['METE_MAE']) / df['METE_MAE']
        #
        # best_idx = diff.idxmin()  # index of the minimum difference
        # worst_idx = diff.idxmax()  # index of the maximum difference
        #
        # best = df.loc[best_idx]
        # worst = df.loc[worst_idx]
        #
        # print(f"Best:\n{best}\n")
        # print(f"Worst:\n{worst}\n")

        row_tf = {
            'r2_dn': df['r2_dn'][0],
            'r2_de': df['r2_de'][0],
            'PlotName': plot
        }
        tf_df.append(row_tf)

        row_stats = do_statistics(df, metric=metric)
        row_stats['PlotName'] = plot
        statistics_df.append(row_stats)

        #scatter_per_year(df, metric, plot)

    # Convert lists to DataFrames
    tf_df = pd.DataFrame(tf_df)
    statistics_df = pd.concat(statistics_df, ignore_index=True)

    make_scatter_plot(metric)
    tf_table = make_transition_function_table()

    print("End")