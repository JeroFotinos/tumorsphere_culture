"""Module for plotting the time evolution of the population numbers."""

from typing import List, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import pandas as pd

from tumorsphere.library.dataframe_generation import average_over_realizations


def plot_single_realization(
    df: pd.DataFrame,
    culture_id: int = None,
    ps: float = None,
    pd: float = None,
    rng_seed: int = None,
    plot_total_cells: bool = True,
    plot_active_cells: bool = True,
    plot_stem_cells: bool = True,
    plot_active_stem_cells: bool = True,
    log: bool = False,
) -> None:
    r"""
    Plot the time evolution of the populations for a given realization.

    You can plot the total cells, active cells, stem cells, and active stem
    cells. If the DataFrame was generated through the SQL databases, you can
    filter by culture_id. If the DataFrame was generated through the `.dat`
    files, you can filter by ps, pd, and rng_seed. If you provide a
    culture_id, the other parameters are ignored.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data as given by `tumorsphere makedf`.
    culture_id : int, optional
        Culture ID to filter the DataFrame. If provided, ps, pd, and rng_seed
        are ignored.
    ps : float, optional
        Probability of stem cell self-renewal.
    pd : float, optional
        Probability of differentiation after giving a DCC.
    rng_seed : int, optional
        Random number generator seed.
    plot_total_cells : bool, default=True
        Whether to plot total cells.
    plot_active_cells : bool, default=True
        Whether to plot active cells.
    plot_stem_cells : bool, default=True
        Whether to plot total stem cell number.
    plot_active_stem_cells : bool, default=True
        Whether to plot active stem cells.
    log : bool, default=False
        Whether to use a logarithmic scale for the y-axis.

    Raises
    ------
    ValueError
        If culture_id is not provided and ps, pd, and rng_seed are not
        specified.
        If no data matches the provided criteria.

    Examples
    --------
    >>> plot_single_realization(df, culture_id=1, \
    ... plot_total_cells=False, log=True)
    >>> plot_single_realization(df, ps=0.6, pd=0.7, \
    ... rng_seed=23, plot_active_cells=False)
    """
    # Filter the DataFrame based on the provided parameters
    if culture_id is not None:
        filtered_df = df[df["culture_id"] == culture_id]
        # Extract ps and pd from the filtered DataFrame
        ps = filtered_df["ps"].iloc[0]
        pd = filtered_df["pd"].iloc[0]
    else:
        if ps is None or pd is None or rng_seed is None:
            raise ValueError(
                (
                    "If culture_id is not provided, "
                    "ps, pd, and rng_seed must be specified."
                )
            )
        filtered_df = df[
            (df["ps"] == ps) & (df["pd"] == pd) & (df["rng_seed"] == rng_seed)
        ]

    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        raise ValueError("No data matches the provided criteria.")

    # Default color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plotting
    plt.figure(figsize=(10, 6))

    if plot_total_cells:
        plt.plot(
            filtered_df["time"],
            filtered_df["total_cells"],
            marker=".",
            linestyle="-",
            label="Total Cells",
            color=default_colors[0],
        )

    if plot_active_cells:
        plt.plot(
            filtered_df["time"],
            filtered_df["active_cells"],
            marker=".",
            linestyle="-",
            label="Active Cells",
            color=default_colors[2],
        )

    if plot_stem_cells:
        plt.plot(
            filtered_df["time"],
            filtered_df["stem_cells"],
            marker=".",
            linestyle="-",
            label="Stem Cells",
            color=default_colors[1],
        )

    if plot_active_stem_cells:
        plt.plot(
            filtered_df["time"],
            filtered_df["active_stem_cells"],
            marker=".",
            linestyle="-",
            label="Active Stem Cells",
            color=default_colors[3],
        )

    plt.xlabel("Time")
    plt.ylabel("Cell Count")
    plt.title(
        "Time Evolution of Cell Populations "
        f"with $p_s = ${ps} and $p_d = ${pd}"
    )
    plt.legend()

    if log:
        plt.yscale("log")

    plt.grid(True)
    plt.show()


def plot_avg_evolution(
    df: pd.DataFrame,
    ps: Union[float, List[float]],
    pd: float,
    plot_total_cells: bool = False,
    plot_active_cells: bool = False,
    plot_stem_cells: bool = False,
    plot_active_stem_cells: bool = True,
    log: bool = False,
) -> None:
    """
    Plot the average time evolution of the populations for given ps and pd.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    ps : float or List[float]
        Probability or list of probabilities of stem cell proliferation.
    pd : float
        Probability of differentiation.
    plot_total_cells : bool, default=True
        Whether to plot total cells.
    plot_active_cells : bool, default=True
        Whether to plot active cells.
    plot_stem_cells : bool, default=True
        Whether to plot stem cells.
    plot_active_stem_cells : bool, default=True
        Whether to plot active stem cells.
    log : bool, default=False
        Whether to use a logarithmic scale for the y-axis.

    Raises
    ------
    ValueError
        If no data matches the provided criteria.

    Examples
    --------
    >>> plot_avg_evolution(df_from_dbs, ps=[0.6, 0.7], pd=0.1)
    """
    # Ensure ps is a list
    if isinstance(ps, float):
        ps = [ps]

    # Filter the DataFrame for the given pd and ps values
    filtered_df = df[(df["pd"] == pd) & (df["ps"].isin(ps))]

    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        raise ValueError("No data matches the provided criteria.")

    # Compute the average over realizations
    avg_df = average_over_realizations(
        filtered_df,
        avg_stem_indicator=False,
        calculate_stem_indicator=False,
    )

    # Default color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plotting
    plt.figure(figsize=(10, 6))

    if len(ps) == 1:
        ps_value = ps[0]
        ps_df = avg_df[avg_df["ps"] == ps_value]

        if plot_total_cells:
            plt.errorbar(
                ps_df["time"],
                ps_df["total_cells"],
                label="Total Cells",
                yerr=ps_df["total_cells_std"],
                color=default_colors[0],
                uplims=True,
                lolims=True,
                marker=".",
                linestyle="-",
            )

        if plot_active_cells:
            plt.errorbar(
                ps_df["time"],
                ps_df["active_cells"],
                yerr=ps_df["active_cells_std"],
                label="Active Cells",
                color=default_colors[2],
                uplims=True,
                lolims=True,
                marker=".",
                linestyle="-",
            )

        if plot_stem_cells:
            plt.errorbar(
                ps_df["time"],
                ps_df["stem_cells"],
                yerr=ps_df["stem_cells_std"],
                label="Stem Cells",
                color=default_colors[1],
                uplims=True,
                lolims=True,
                marker=".",
                linestyle="-",
            )

        if plot_active_stem_cells:
            plt.errorbar(
                ps_df["time"],
                ps_df["active_stem_cells"],
                yerr=ps_df["active_stem_cells_std"],
                label="Active Stem Cells",
                color=default_colors[3],
                uplims=True,
                lolims=True,
                marker=".",
                linestyle="-",
            )
    else:
        cmap = cm.get_cmap("viridis", len(ps))
        for idx, ps_value in enumerate(ps):
            ps_df = avg_df[avg_df["ps"] == ps_value]
            # color = cmap(idx)
            # Avoid extreme colors:
            # color = cmap(0.1 + 0.37 * (idx / (len(ps) - 1)))
            # color = cmap(2 * (ps_value) - 1)
            # color = cmap(ps_value)
            color = cmap(idx / (len(ps) - 1))

            if plot_total_cells:
                plt.errorbar(
                    ps_df["time"],
                    ps_df["total_cells"],
                    yerr=ps_df["total_cells_std"],
                    label=f"Total Cells $p_s=${ps_value}",
                    color=color,
                    uplims=True,
                    lolims=True,
                    marker=".",
                    linestyle="-",
                )

            if plot_active_cells:
                plt.errorbar(
                    ps_df["time"],
                    ps_df["active_cells"],
                    yerr=ps_df["active_cells_std"],
                    label=f"Active Cells $p_s=${ps_value}",
                    color=color,
                    uplims=True,
                    lolims=True,
                    marker=".",
                    linestyle="-",
                )

            if plot_stem_cells:
                plt.errorbar(
                    ps_df["time"],
                    ps_df["stem_cells"],
                    yerr=ps_df["stem_cells_std"],
                    label=f"Stem Cells $p_s=${ps_value}",
                    color=color,
                    uplims=True,
                    lolims=True,
                    marker=".",
                    linestyle="-",
                )

            if plot_active_stem_cells:
                plt.errorbar(
                    ps_df["time"],
                    ps_df["active_stem_cells"],
                    yerr=ps_df["active_stem_cells_std"],
                    label=f"Active Stem Cells $p_s=${ps_value}",
                    color=color,
                    uplims=True,
                    lolims=True,
                    marker=".",
                    linestyle="-",
                )

    plt.xlabel("Time")
    plt.ylabel("Cell Count")
    if len(ps) == 1:
        plt.title(
            (
                "Average Time Evolution of Cell Populations "
                f"with $p_s = ${ps_value} and $p_d = ${pd}"
            )
        )
    else:
        plt.title(
            f"Average Time Evolution of Cell Populations with $p_d = ${pd}"
        )
    plt.legend()

    if log:
        plt.yscale("log")

    plt.grid(True)
    plt.show()
