import seaborn.objects as so
import numpy as np
import pandas as pd
import seaborn as sns
from utils.general import compose_file_name
from enums.enums import FeatureEnum as FE
from pathlib import Path
import sklearn.manifold as manifold
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc, ticker
import os


_CONTEXT = "paper"
sns.set_context(_CONTEXT)
_STYLE = "darkgrid"
_RC = {"axes.facecolor": ".93", "axes.spines.right": False, "axes.spines.top": False}
sns.set_style(_STYLE, _RC)


def kde_facet_plot(
    data: pd.DataFrame,
    feature: str,
    color: str,
    facet: str,
    x_lim: tuple = (-0.05, 1),
    out_dir: Path = None,
    out_type="svg",
):
    """Multi-purpose plot for displaying the distribution of a continuous feature with multiple colors and facets.

    Parameters
    ----------
    data : pd.DataFrame
        The data.
    feature : str
        Column name. The feature for which the distribution is plotted.
    color : str
        Column name. Must be categorical. Partitions of "feature" plotted on the same graph.
    facet : str
        Column name. Must be categorical. Partitions of "feature" plotted side-by-side.
    x_lim : tuple[float]
        Sets the plot range for "feature". Overwritten if feature values exceed 1.
    out_dir : Path
        Path to store the figure.
    out_type : str
        Output type.

    """
    x_lim_dict = {}
    if data[feature].abs().max() <= 1:
        x_lim_dict["x"] = x_lim
    p = (
        so.Plot(data, x=feature, color=color)
        .facet(col=facet)
        .add(so.Line(), so.KDE(common_norm=True, bw_adjust=0.5))
        .label(x=feature, y="density", color=color)
        .layout(size=(10, 5))
        .limit(**x_lim_dict)
    )
    if out_dir:
        file_name = compose_file_name(
            "kde", f"x-{feature}", f"facet-{facet}", f"color-{color}"
        )
        file_name += "." + out_type
        out_path = out_dir / file_name
        p.save(out_path, bbox_inches="tight")
    return p


def hist_facet_plot(
    data: pd.DataFrame,
    feature: str,
    color: str,
    facet: str,
    x_lim: tuple = (-0.05, 1),
    out_dir: Path = None,
    out_type="svg",
):
    """Multi-purpose plot for displaying the histogram of a continuous feature with multiple colors and facets.

    Parameters
    ----------
    data : pd.DataFrame
        The data.
    feature : str
        Column name. The feature for which the distribution is plotted.
    color : str
        Column name. Must be categorical. Partitions of "feature" plotted on the same graph.
    facet : str
        Column name. Must be categorical. Partitions of "feature" plotted side-by-side.
    x_lim : tuple[float]
        Sets the plot range for "feature". Overwritten if feature values exceed 1.
    out_dir : Path
        Path to store the figure.
    out_type : str
        Output type.

    """
    x_lim_dict = {}
    if data[feature].abs().max() <= 1:
        x_lim_dict["x"] = x_lim
    p = (
        so.Plot(data, x=feature, color=color)
        .facet(col=facet)
        .add(
            so.Bar(alpha=1),
            so.Hist(stat="count", bins=8, common_norm=True),
            so.Dodge(),
        )
        .label(x=feature, y="count", color=color)
        .layout(size=(10, 5))
        .limit(**x_lim_dict)
    )
    if out_dir:
        file_name = compose_file_name(
            "hist", f"x-{feature}", f"facet-{facet}", f"color-{color}"
        )
        file_name += "." + out_type
        out_path = out_dir / file_name
        p.save(out_path, bbox_inches="tight")
    return p


def mds_tool_scatterplot(
    dist_mat: pd.DataFrame,
    dataset: str,
    color: pd.Series = None,
    metric_name: FE = FE.HOMOLOGY_POS_DIST,
    out_dir: Path = None,
    out_type="svg",
    method: str = "tsne",
):
    if method == "tsne":
        tsne = manifold.TSNE(
            n_components=2,
            perplexity=10,
            early_exaggeration=5,
            init="random",
            # method="exact",
            metric="precomputed",
            random_state=0,
        )
        points = tsne.fit_transform(dist_mat)
        variance_ratios = np.array([0, 0])

    # mds = manifold.MDS(
    #     n_components=2,
    #     dissimilarity="precomputed",
    #     normalized_stress="auto",
    #     metric=False,
    #     random_state=0,
    # )
    # points = mds.fit_transform(dist_mat)
    # variance_ratios = np.array([0, 0])

    elif method == "metric":
        u, s, _ = np.linalg.svd(dist_mat, full_matrices=True)
        points = u[:, :2]
        variance_ratios = s / s.sum()
        print(dataset, variance_ratios[:5].round(2))

    else:
        raise ValueError(f"Unknown method '{method}'")

    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    points = (points - min_vals) / (max_vals - min_vals)
    # explained_var = mds.explained_variance_ratio_
    explained_var = np.array(variance_ratios[:2])
    pca_df = pd.DataFrame(points, columns=["PC_1", "PC_2"])
    tools = dist_mat.index.to_series(name="tool").reset_index(drop=True)
    if "." in dataset:
        dataset = dataset.split(".")[0]
    # title = f"Multidimensional Scaling for Dataset {dataset}."

    sns.set_style("dark", _RC)
    fig = plt.figure(figsize=(5, 4))
    cmap = None
    color_norm = None
    if color is not None:
        cmap_name = "viridis"
        cmap = sns.color_palette(cmap_name, as_cmap=True)
        color_norm = colors.Normalize(color.min() * 0.9, color.max() * 1.1)
    ax = sns.scatterplot(
        pca_df,
        x="PC_1",
        y="PC_2",
        hue=tools,
        s=100,
        style=tools,
        alpha=0.7,
        palette=cmap,
        hue_norm=color_norm,
    )

    sns.set_style(_STYLE, _RC)
    if method == "metric":
        ax.set_xlabel(f"1st component: {explained_var[0]:.0%}")
        ax.set_ylabel(f"2nd component: {explained_var[1]:.0%}")
        var_text = f"$\\sum = ${explained_var.sum():.0%}"
        ax.text(-0.2, -0.15, var_text)
    else:
        ax.set_xlabel("1st component")
        ax.set_ylabel("2nd component")
    ax.set_xlim(-0.1, 1.5)
    # ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Remove color handles from legend
    # h, l = ax.get_legend_handles_labels()
    # num_tools = len(tools.unique())
    # ax.legend(h[-num_tools:], l[-num_tools:])

    if color is not None:
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=color_norm)
        sm.set_array(color)
        cb = fig.colorbar(sm, ax=ax)
        tick_locator = ticker.MaxNLocator(nbins=7, steps=[5])
        cb.locator = tick_locator
        cb.update_ticks()

    if out_dir:
        file_name = compose_file_name(
            "mds", f"dataset-{dataset}", f"metric-{metric_name}"
        )
        file_name += "." + out_type
        out_path = out_dir / file_name
        fig.savefig(out_path, bbox_inches="tight")
    return fig
