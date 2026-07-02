import ast
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import wilcoxon


# --- Format numbers ---
def format_value(x, col=None):
    if isinstance(x, (int, float)):
        if col == "slack_weight":
            return f"{x:.1e}"  # scientific notation, 1 decimal
        else:
            return f"{x:.2f}".rstrip("0").rstrip(".")  # normal 2-decimal format
    return x

def metrics_per_slack_weight(df, frac):
    # Calculate group averages
    df_mean = df.groupby(["slack_weight", "iter"])[
        ["METimE_AIC", "METimE_MAE", "METimE_RMSE", "METE_AIC", "METE_MAE", "METE_RMSE"]
    ].mean().reset_index()

    # Create figure with shared x-axis
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # Colormap for consistent census colors
    cmap = plt.get_cmap("tab10", df["iter"].nunique())
    census_colors = {c: cmap(i) for i, c in enumerate(sorted(df["iter"].unique()))}

    # --- Top subplot: MAE ---
    for iter, group in df.groupby("iter"):
        color = census_colors[iter]
        axes[0].plot(group["slack_weight"], group["METimE_MAE"], color=color, label=f"Iteration {iter}")
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
    for iter, group in df.groupby("iter"):
        color = census_colors[iter]
        axes[1].plot(group["slack_weight"], group["METimE_RMSE"], color=color, label=f"Iteration {iter}")
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
    for iter, group in df.groupby("iter"):
        color = census_colors[iter]
        axes[2].plot(group["slack_weight"], group["METimE_AIC"], color=color, label=f"Iteration {iter}")
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

def fill_latex_table(df):
    # Apply formatting column by column
    for col in df.columns:
        df[col] = df[col].apply(lambda x: format_value(x, col))

    # --- Reorder columns (added slack_weight after quad and census) ---
    cols = [
        "frac", "r2_dn", "r2_de", "METE_AIC", "METE_MAE", "METE_RMSE",
        "METimE_AIC", "METimE_MAE", "METimE_RMSE"
    ]
    df = df[cols]

    # --- Convert to LaTeX without headers ---
    latex_table = df.to_latex(
        index=False,
        header=False,
        column_format="c|cc|ccc|ccc",
        escape=False
    )

    # --- Build custom header (added slack_weight) ---
    custom_header = (
        "\\toprule\n"
        " &  & &  \\multicolumn{3}{c|}{METE} & \\multicolumn{3}{c}{METimE} \\\\\n"
        "Fraction removed & r2_dn & r2_de & AIC & MAE & RMSE & AIC & MAE & RMSE \\\\\n"
        "\\midrule\n"
    )

    # --- Insert header ---
    latex_table = latex_table.replace("\\toprule", custom_header, 1)

    # --- Save to file ---
    with open("table.tex", "w") as f:
        f.write(latex_table)

    print(latex_table)

def select_best_slack_weight(df, metric="MAE"):
    results = []

    for frac in df['frac'].unique():
        for iter in df['iter'].unique():
            df_subset = df[(df['frac'] == frac) & (df['iter'] == iter)]

            # find row that minimizes MAE
            best_idx = df_subset[f'METimE_{metric}'].idxmin()
            best_row = df_subset.loc[best_idx]
            print("Best weight: ", best_row['slack_weight'])

            results.append(best_row)

    # return a DataFrame of the selected best rows
    return pd.DataFrame(results).reset_index(drop=True)

def print_additional_metrics(df):
    n = len(df)

    # Calculate how often METimE outperforms METE
    better_AIC   = (df["METE_AIC"]   > df["METimE_AIC"]).sum()   / n * 100
    better_MAE   = (df["METE_MAE"]   > df["METimE_MAE"]).sum()   / n * 100
    better_RMSE  = (df["METE_RMSE"]  > df["METimE_RMSE"]).sum()  / n * 100
    better_NS    = (df["METE_error_N/S"]  > df["METimE_error_N/S"]).sum() / n * 100
    better_ES    = (df["METE_error_E/S"]  > df["METimE_error_E/S"]).sum() / n * 100
    better_NoverS= (df["METE_error_dN/S"] > df["METimE_error_dN/S"]).sum() / n * 100
    better_EoverS= (df["METE_error_dE/S"] > df["METimE_error_dE/S"]).sum() / n * 100

    # Calculate how often they're equal
    equal_AIC   = (df["METE_AIC"]   == df["METimE_AIC"]).sum()   / n * 100
    equal_MAE   = (df["METE_MAE"]   == df["METimE_MAE"]).sum()   / n * 100
    equal_RMSE  = (df["METE_RMSE"]  == df["METimE_RMSE"]).sum()  / n * 100
    equal_NS    = (df["METE_error_N/S"]  == df["METimE_error_N/S"]).sum() / n * 100
    equal_ES    = (df["METE_error_E/S"]  == df["METimE_error_E/S"]).sum() / n * 100
    equal_NoverS= (df["METE_error_dN/S"] == df["METimE_error_dN/S"]).sum() / n * 100
    equal_EoverS= (df["METE_error_dE/S"] == df["METimE_error_dE/S"]).sum() / n * 100

    # Build summary table with LaTeX-friendly labels
    summary = pd.DataFrame({
        "Metric": [
            "AIC", "MAE", "RMSE", r"$N/S$ error", r"$E/S$ error",
            r"$\Delta N/S$ error", r"$\Delta E/S$ error"
        ],
        "METimE better than METE (\\%)": [
            better_AIC, better_MAE, better_RMSE, better_NS, better_ES, better_NoverS, better_EoverS
        ],
        "METE as good as METimE (\\%)": [
            equal_AIC, equal_MAE, equal_RMSE, equal_NS, equal_ES, equal_NoverS, equal_EoverS
        ]
    })

    # Format as LaTeX table
    latex_table = summary.to_latex(
        index=False,
        escape=False,  # keep LaTeX math symbols
        float_format="%.2f"
    )

    print(latex_table)

def how_much_difference(df):
    #sns.set_theme(style="white")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Colors
    blueish = "#67a9cf"
    greyish = "#4c4c4c"
    orangy = "#ef8a62"

    # Ensure numeric
    for col in ['METE_AIC', 'METimE_AIC', 'METE_MAE', 'METimE_MAE', 'METE_RMSE', 'METimE_RMSE', 'METE_error_N/S', 'METimE_error_N/S', 'METE_error_E/S', 'METimE_error_E/S', 'METE_error_dN/S', 'METimE_error_dN/S', 'METE_error_dE/S', 'METimE_error_dE/S']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute differences
    df_diff = pd.DataFrame({
        'AIC': df['METE_AIC'] - df['METimE_AIC'],
        'MAE': df['METE_MAE'] - df['METimE_MAE'],
        'RMSE': df['METE_RMSE'] - df['METimE_RMSE'],
        'N/S error': df['METE_error_N/S'] - df['METimE_error_N/S'],
        'E/S error': df['METE_error_E/S'] - df['METimE_error_E/S'],
        'dN/S error': df['METE_error_dN/S'] - df['METimE_error_dN/S'],
        'dE/S error': df['METE_error_dE/S'] - df['METimE_error_dE/S']
    })

    metrics = ['AIC', 'MAE', 'RMSE', 'N/S error', 'E/S error', 'dN/S error', 'dE/S error']

    fig, axes = plt.subplots(1, 7, figsize=(24, 5), sharey=False)

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        values = df_diff[metric].dropna()

        # Violin plot without inner box
        sns.violinplot(
            y=values, ax=ax, color=blueish, alpha=0.8,
            inner=None, bw_adjust=0.5, cut=0, zorder=1
        )

        # Overlay custom boxplot (smaller width, rounded, black fill, white median)
        sns.boxplot(
            y=values, ax=ax, width=0.1, showcaps=False, showfliers=False,
            boxprops=dict(facecolor=greyish, edgecolor=greyish, linewidth=1.2),
            whiskerprops=dict(color=greyish, linewidth=1.0),
            capprops=dict(color=greyish, linewidth=1.0),
            medianprops=dict(color="white", linewidth=3),
            flierprops=dict(markerfacecolor=greyish, markersize=5, alpha=0.5)
        )

        # Strong horizontal line at 0
        ax.axhline(0, color=orangy, linewidth=4, linestyle="-", zorder=0)

        # Compute percentage above 0
        total = len(values)
        above = (values > 0).sum()
        perc_above = above / total * 100 if total > 0 else 0

        # Annotate percentages
        ax.text(0.5, 0.98, f"{perc_above:.1f}% > 0",
                ha='center', va='top', transform=ax.transAxes,
                fontsize=14)

        ax.set_title(metric, fontsize=16)

        if i == 0:
            ax.set_ylabel("Difference (METE - METimE)", fontsize=14)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis='both', which='major', labelsize=12)

        # # Inset boxplot on the right, centered vertically
        # inset_ax = inset_axes(ax, width="25%", height="40%", loc="lower right",
        #                       borderpad=1.2)
        #
        # sns.boxplot(
        #     y=values, ax=inset_ax, width=0.2, showcaps=True, showfliers=False,
        #     boxprops=dict(facecolor=greyish, edgecolor="black", linewidth=1.2),
        #     whiskerprops=dict(color=greyish, linewidth=1.0),
        #     capprops=dict(color=greyish, linewidth=1.0),
        #     medianprops=dict(color="white", linewidth=3.0)
        # )
        #
        # inset_ax.axhline(0, color=orangy, linewidth=4, linestyle="-")
        # inset_ax.set_xticks([])
        # inset_ax.set_xlabel("")
        # inset_ax.set_ylabel("")
        # inset_ax.tick_params(axis='y', labelsize=8)

        ylim_min, ylim_max = ax.get_ylim()
        ylim_max *= 1.80  # scale max by 10%
        ax.set_ylim(ylim_min, ylim_max)

    plt.tight_layout()
    plt.show()

def cleaner_look(df):
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

    # Compute differences
    df_diff = pd.DataFrame({
        'frac': df['frac'],
        'iter': df['iter'],
        'AIC': (df['METE_AIC'] - df['METimE_AIC']) / df['METE_AIC'],
        'MAE': (df['METE_MAE'] - df['METimE_MAE']) / df['METE_MAE'],
        'RMSE': (df['METE_RMSE'] - df['METimE_RMSE']) / df['METE_RMSE'],
        'N/S error': (df['METE_error_N/S'] - df['METimE_error_N/S']) / df['METE_error_N/S'],
        'E/S error': (df['METE_error_E/S'] - df['METimE_error_E/S']) / df['METE_error_E/S']
    })

    metrics = ['AIC', 'MAE', 'RMSE', 'N/S error', 'E/S error']

    fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=False)

    for i, (ax, metric) in enumerate(zip(axes, metrics)):

        # Use boxplot (handles repetitions via 'iter')
        sns.boxplot(
            x='frac',
            y=metric,
            data=df_diff,
            palette='Set2',
            hue='frac',
            legend=False,
            ax=ax,
            showfliers=False,
            linewidth=2
        )

        # Strong horizontal line at 0
        ax.axhline(0, color=greyish, linewidth=3, linestyle="-", zorder=1)

        if i == 0:
            ax.set_ylabel("Relative difference", fontsize=18, linespacing=1.5)
        else:
            ax.set_ylabel("")

        ax.set_xlabel("")
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(metric, fontsize=18)

    # Figure-wide x-label
    fig.text(0.5, 0.05, "Fraction of population removed", ha='center', fontsize=18)
    fig.text(0.5, 0.95, "Metric", ha='center', fontsize=22)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.show()

def cleaner_look_single(df):
    dark_greyish = "#4c4c4c"
    greyish = "#707070"

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
        'frac': df['frac'],
        'iter': df['iter'],
        'AIC': (df['METE_AIC'] - df['METimE_AIC']) / df['METE_AIC'],
        'MAE': (df['METE_MAE'] - df['METimE_MAE']) / df['METE_MAE'],
        'RMSE': (df['METE_RMSE'] - df['METimE_RMSE']) / df['METE_RMSE'],
        'N/S error': (df['METE_error_N/S'] - df['METimE_error_N/S']) / df['METE_error_N/S'],
        'E/S error': (df['METE_error_E/S'] - df['METimE_error_E/S']) / df['METE_error_E/S']
    })

    # ➡️ Melt to long format
    df_long = df_diff.melt(
        id_vars=['frac', 'iter'],
        value_vars=['MAE', 'RMSE', 'N/S error', 'E/S error'],
        var_name='Metric',
        value_name='Relative difference'
    )

    # ✅ One big boxplot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        x='Metric',
        y='Relative difference',
        hue='frac',
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
                    ncol=5,
                    title="Fraction of equilibrium population removed",
                    frameon=False)

    # Strong horizontal line at 0
    ax.axhline(0, color=greyish, linewidth=2, linestyle="--", zorder=1)

    ax.set_xlabel("")
    ax.set_ylabel("Relative difference \n (METE - METimE) / METE", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_title("Comparison of Metrics by Fraction Removed", fontsize=20)

    # # plt.legend(title="Fraction of \n population removed", fontsize=12, title_fontsize=13)
    # sns.move_legend(ax, "lower center",
    #                 bbox_to_anchor=(.5, 1),
    #                 ncol=5,
    #                 title="Fraction of equilibrium population removed",
    #                 frameon=False)

    plt.tight_layout()

    plt.savefig(
        "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/simulated_BCI/new_simulated_BCI_boxplot.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )
    plt.show()

    # ➡️ Prepare data for AIC violin plot
    df_aic = pd.melt(
        df,
        id_vars=['frac', 'iter'],
        value_vars=['METE_AIC', 'METimE_AIC'],
        var_name='Model',
        value_name='AIC'
    )

    # ✅ Rename for cleaner legend/labels
    df_aic['Method'] = df_aic['Model'].replace({
        'METE_AIC': 'METE',
        'METimE_AIC': 'METimE'
    })

    g = sns.catplot(
        y="AIC",
        col="frac",
        hue="Method",
        data=df_aic,
        kind="box",
        linewidth=1.5,
        showfliers=False,
        sharey=False,
        height=6,
        aspect=0.3,
        width=.7
    )

    # Remove facet titles
    g.set_titles("")
    handles, labels = g.axes[0, 0].get_legend_handles_labels()

    # Add fraction values as custom x-labels under each subplot
    for ax, frac_value in zip(g.axes.flat, sorted(df_aic["frac"].unique())):
        ax.set_xlabel(f"{frac_value:.1f}", fontsize=14)
        ax.tick_params(axis='y', labelsize=12)

    # Global y-axis label
    g.set_ylabels("AIC", fontsize=18)

    #Add legend above the plots, 2 columns
    g._legend.remove()
    g.add_legend(
        handles=handles,
        labels=labels,
        title="MaxEnt method",
        frameon=True,
        fontsize=16,
        title_fontsize=18,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98)  # moves it above all subplots
    )
    g._legend.get_title().set_fontsize(18)

    sns.despine()

    # Adjust margins to fit x-axis labels and legend
    g.figure.subplots_adjust(
        left=0.1,
        right=0.95,
        top=0.83,  # leave space for the legend at top
        bottom=0.12,
        wspace=0.15
    )

    # Global x-axis label
    g.figure.text(
        0.5, 0.01,
        "Fraction of equilibrium population removed",
        ha="center",
        fontsize=18
    )

    # Save figure
    g.figure.savefig(
        "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/simulated_BCI/new_simulated_BCI_AIC_violin.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )

    plt.show()

def scatterplot(df):
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

    # Rectangular subplots (wider than tall)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=False)

    # We'll capture handles/labels from the FIRST plot for the global legend
    handles, labels = None, None

    for ax, metric, METE_metric, METimE_metric in zip(axes, metrics, METE_metrics, METimE_metrics):

        # Scatter plot
        sns.scatterplot(
            x=METE_metric,
            y=METimE_metric,
            data=df,
            hue="frac",
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

        #if metric in ["MAE", "RMSE"]:
            # # Create inset inside this subplot
            # inset = inset_axes(ax, width="40%", height="40%", loc="lower right")
            # sns.scatterplot(
            #     x=METE_metric, y=METimE_metric, data=df,
            #     hue="frac", palette="Set2",
            #     ax=inset, s=40, alpha=0.8,
            #     edgecolor=greyish, linewidth=0.8,
            #     legend=False
            # )
            #
            # # Zoom limits
            # if metric == "MAE":
            #     inset.set_xlim(0, 6)
            #     inset.set_ylim(0, 6)
            #     inset.plot([0, 6], [0, 6], '--', color=greyish, lw=1)
            # else:  # RMSE
            #     inset.set_xlim(0, 20)  # adjust if needed
            #     inset.set_ylim(0, 20)
            #     inset.plot([0, 20], [0, 20], '--', color=greyish, lw=1)
            #
            # inset.tick_params(axis="both", labelsize=8)
            # inset.set_xlabel("")
            # inset.set_ylabel("")

        # Styling
        ax.set_xlabel("METE", fontsize=12)
        ax.set_ylabel("METimE", fontsize=12)
        ax.set_title(metric, fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=10)

    # Remove per-axes legends
    for ax in axes:
        ax.get_legend().remove()

    # Global legend underneath
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=12)

    fig.text(0.5, 0.95, "METE vs METimE Metrics", ha="center", fontsize=20)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])

    plt.savefig(
        "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/simulated_BCI/simulated_BCI_scatterplot.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )

    plt.show()

def transition_functions(df):
    dark_greyish = "#4c4c4c"
    greyish = "#707070"

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Ensure numeric
    df['r2_dn'] = pd.to_numeric(df['r2_dn'], errors='coerce')
    df['r2_de'] = pd.to_numeric(df['r2_de'], errors='coerce')

    # Create two separate DataFrames
    df_dn = pd.DataFrame({
        'R^2': df['r2_dn'],
        'frac': df['frac']
    })
    df_de = pd.DataFrame({
        'R^2': df['r2_de'],
        'frac': df['frac']
    })

    # Set up side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Panel 1: r2_dn
    sns.boxplot(
        x='frac',
        y='R^2',
        data=df_dn,
        palette='Set2',
        ax=axes[0],
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
    axes[0].set_title(r"$f \approx \Delta n$", fontsize=20)
    axes[0].set_xlabel("Fraction of equilibrium population removed", fontsize=18)
    axes[0].set_ylabel("Coefficient of determination (R²)", fontsize=18)
    axes[0].tick_params(axis='both', labelsize=14)

    # Panel 2: r2_de
    sns.boxplot(
        x='frac',
        y='R^2',
        data=df_de,
        palette='Set2',
        ax=axes[1],
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
    axes[1].set_title(r"$h \approx \Delta \overline{\varepsilon}$", fontsize=20)
    axes[1].set_xlabel("Fraction of equilibrium population removed", fontsize=18)
    axes[1].set_ylabel("Coefficient of determination (R²)", fontsize=18)
    axes[1].tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    plt.savefig(
        "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/simulated_BCI/simulated_BCI_transition_functions.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )
    plt.show()

def do_statistics(df):
    results = []  # Store test results

    for frac in df['frac'].unique():
        df_model = df[(df['frac'] == frac)]

        # Ensure we have both columns
        if not {'METE_MAE', 'METimE_MAE'}.issubset(df_model.columns):
            raise ValueError("DataFrame must contain 'METE_MAE' and 'METimE_MAE' columns.")

        diff = df_model['METE_MAE'] - df_model['METimE_MAE']

        wilcoxon_res = wilcoxon(df_model['METE_MAE'], df_model['METimE_MAE'], method="asymptotic")
        p_val_wilcoxon = wilcoxon_res.pvalue
        z_val_wilcoxon = wilcoxon_res.zstatistic

        # Collect results
        results.append({
            'frac': frac,
            'wilcoxon_p': p_val_wilcoxon,
            'wilcoxon_z': z_val_wilcoxon,
            'median_METE': np.median(df_model['METE_MAE']),
            'median_METimE': np.median(df_model['METimE_MAE']),
        })

    results_df = pd.DataFrame(results)

    results_df = results_df[['frac', 'median_METE', 'median_METimE', 'wilcoxon_p', 'wilcoxon_z']]

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


def get_microscopic_variance():
    dark_greyish = "#4c4c4c"
    greyish = "#707070"

    # Load CSV
    df = pd.read_csv("variance simulated BCI.csv",
                     names=['dn', 'de'])

    # Add 'frac' column: 0.0, 0.2, 0.4, 0.6, 0.8 repeated 20 times each
    frac_values = np.repeat([0.0, 0.2, 0.4, 0.6, 0.8], 20)
    df['dn'] = df['dn'].apply(ast.literal_eval)
    df['de'] = df['de'].apply(ast.literal_eval)
    df['frac'] = frac_values[:len(df)]  # ensure it matches the dataframe length

    # Set up 2x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Step 1: explode dn lists into long format
    df_long = df[['dn', 'frac']].explode('dn')
    df_long['dn'] = pd.to_numeric(df_long['dn'])

    # Upper-left: mean dn
    # sns.boxplot(data=df_long, x='frac', y='dn', hue='frac', ax=axes[0], palette='Set2', showfliers=True,
    #             medianprops={
    #                 "color": dark_greyish,
    #                 "linewidth": 3
    #             },
    #             flierprops={
    #                 "markerfacecolor": greyish,
    #                 "markeredgecolor": greyish,
    #                 "alpha": 0.1,
    #                 "markersize": 4,
    #                 "linestyle": "none"
    #             }
    #             )

    sns.violinplot(data=df_long, x='frac', y='dn', hue='frac', ax=axes[0], palette='Set2',
                   inner = "box",
                   density_norm="count")

    sns.boxplot(
        data=df_long, x='frac', y='dn', hue='frac', ax=axes[0],
        showcaps=False,
        whiskerprops={'visible': False},
        showfliers=False,
        boxprops={'visible': False},
        medianprops={'color': 'white',
                     'linewidth': 3},
        width=0.3
    )

    axes[0].set_xlabel("Fraction of equilibrium population removed", fontsize=18)
    axes[0].set_ylabel(r"$\Delta n$ (log scale)", fontsize=18)
    axes[0].tick_params(axis='both', labelsize=14)

    # Make the median lines thicker
    for line in axes[0].lines:
        line.set_linewidth(5)  # adjust thickness here

    # Add white dots for the mean
    grouped_means = df_long.groupby("frac")["dn"].mean().reset_index()

    # If you have a hue, you'll need to separate means by hue as well
    # Example (if hue='frac' same as x, then no need to separate)
    for i, mean in enumerate(grouped_means["dn"]):
        axes[0].scatter(i, mean, color="white", s=100, zorder=3, edgecolor="black")

    # Swarmplot overlay
    # sns.swarmplot(
    #     data=df_long, x='frac', y='dn', hue='frac', ax=axes[0], palette='Set2',
    #     dodge=True, alpha=0.5, size=3
    # )

    #axes[0].set_title('Δn')

    df_long = df[['de', 'frac']].explode('de')
    df_long['de'] = pd.to_numeric(df_long['de'])

    # # Upper-right: mean de
    # sns.boxplot(data=df_long, x='frac', y='de', hue='frac', ax=axes[1], palette='plasma', showfliers=True)
    # axes[1].set_ylabel('Δe', fontsize=18)
    # axes[1].set_xlabel('Fraction of population removed', fontsize=16)
    axes[0].set_yscale('symlog')

    sns.violinplot(data=df_long, x='frac', y='de', hue='frac', ax=axes[1], palette='Set2',
                   inner="box",
                   density_norm="count")

    sns.boxplot(
        data=df_long, x='frac', y='de', hue='frac', ax=axes[1],
        showcaps=False,
        whiskerprops={'visible': False},
        showfliers=False,
        boxprops={'visible': False},
        medianprops={'color': 'white', 'linewidth': 3},
        width=0.3
    )

    axes[1].set_yscale('symlog')
    axes[1].set_xlabel("Fraction of equilibrium population removed", fontsize=18)
    axes[1].set_ylabel(r"$\Delta \overline{\varepsilon}$ (log scale)", fontsize=18)
    axes[1].tick_params(axis='both', labelsize=14)

    # Make the median lines thicker
    for line in axes[1].lines:
        line.set_linewidth(5)  # adjust thickness here

    # Add white dots for the mean
    grouped_means = df_long.groupby("frac")["de"].mean().reset_index()

    # If you have a hue, you'll need to separate means by hue as well
    # Example (if hue='frac' same as x, then no need to separate)
    for i, mean in enumerate(grouped_means["de"]):
        axes[1].scatter(i, mean, color="white", s=100, zorder=3, edgecolor="black")

    # Swarmplot overlay
    # sns.swarmplot(
    #     data=df_long, x='frac', y='de', hue='frac', ax=axes[1], palette='Set2',
    #     dodge=True, alpha=0.5, size=3
    # )
    # axes[0].set_title('Δn')
    # axes[0].set_yscale('symlog')

    # Adjust legends
    for ax in axes.flat:
        ax.legend_.remove()  # remove individual legends

    # Add a single legend for the whole figure
    #handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, title='Fraction of population removed', loc='upper center', ncol=5, frameon=False)

    plt.tight_layout()
    plt.savefig(
        "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/simulated_BCI/simulated_BCI_variance.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )
    plt.show()




def summarize_results_latex_by_frac(df: pd.DataFrame) -> str:
    """
    Compute summary stats grouped by 'frac' and return a LaTeX table.

    Expected columns in df:
        'frac', 'METE_MAE', 'METE_RMSE', 'METE_NS', 'METE_ES',
        'METimE_MAE', 'METimE_RMSE', 'METimE_NS', 'METimE_ES'
    """
    # 1️⃣ Select best slack weight per frac if needed
    df = select_best_slack_weight(df, 'MAE')

    metrics = ['MAE', 'RMSE', 'error_N/S', 'error_E/S']
    table_rows = []

    for frac_value, df_frac in df.groupby('frac'):
        # Outliers for METimE_MAE
        diff_ratio = (df_frac['METE_MAE'] - df_frac['METimE_MAE']) / df_frac['METE_MAE']
        q1 = diff_ratio.quantile(0.25)
        iqr = diff_ratio.quantile(0.75) - q1
        lower_bound = q1 - 1.5 * iqr
        outliers = diff_ratio < lower_bound
        pct_outliers = 100 * outliers.mean()  # percentage of 20 iterations

        for m in metrics:
            mete = df_frac[f"METE_{m}"]
            metime = df_frac[f"METimE_{m}"]
            table_rows.append([
                frac_value, m,
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
        "Frac & Metric & METE Max & METimE Max & METimE MAE Outliers (\\%) \\\\\n"
        "\\midrule\n"
    )

    body = "\n".join(
        f"{frac} & {m} & {mete_max} & {metime_max} & {outliers} \\\\"
        for frac, m, mete_max, metime_max, outliers in table_rows
    )

    footer = "\\bottomrule\n\\end{tabular}\n\\caption{Summary stats by frac, showing max values and METimE MAE outliers.}\n\\label{tab:mete_metime_by_frac}\n\\end{table}"

    return header + body + "\n" + footer




if __name__ == "__main__":
    # Load data
    path = 'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/simulated_BCI/new_sBCI_results_29_06.csv'
    df = pd.read_csv(path)

    # # For each fraction, plot the AIC, MAE and RMSE per slack_weight
    # for frac in df['frac'].unique():
    #     df_quad = df[df['frac'] == frac]
    #     metrics_per_slack_weight(df_quad, frac)

    # Select best slack weight
    df = select_best_slack_weight(df, 'MAE')

    # # Remove the one outlier
    # # Find the one with extreme METimE_MAE
    # outliers = df[df['METimE_MAE'] - df['METE_MAE'] > 100]
    # print(outliers)
    # df = df[df['METimE_MAE'] - df['METE_MAE'] <= 100]

    # # Find the best and worst METimE_MAE
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

    # fill_latex_table(df)
    # print_additional_metrics(df)

    cleaner_look_single(df)
    #
    scatterplot(df)
    #
    transition_functions(df)

    do_statistics(df)

    # get_microscopic_variance_points(df)

    latex_table = summarize_results_latex_by_frac(df)
    print(latex_table)

