"""Percolation-oriented post-processing of the simulation data.

This script contains functions to process the output data from the simulations
and create plots and heatmaps. Direct execution on this script will trigger
this post-processing.

There is a boolean variable `db_files` that should be set to `False` if you
want to process the old `.dat` files, and `True` if you want to process the
standard `.db` merged database. If `db_files=False`, an additional step is
used to add the time zero points, because the `.dat` files do not have these.
"""

from tumorsphere.library.dataframe_generation import (
    average_over_realizations,
)

from typing import List, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.special import erf


def plot_p_infty_vs_time(
    mean_df: pd.DataFrame,
    ps: Union[float, List[float]],
    pd: float,
    log: bool = False,
) -> None:
    """
    Plot p_infty as a function of time for given ps values and pd.

    Parameters
    ----------
    mean_df : pd.DataFrame
        DataFrame containing the data, in the form output by
        `average_over_realizations` (it should include the
        'active_stem_cells_indicator' column).
    ps : float or List[float]
        Probability or list of probabilities of stem cell proliferation.
    pd : float
        Probability of differentiation.
    log : bool, default=False
        Whether to use a logarithmic scale for the y-axis.

    Raises
    ------
    ValueError
        If no data matches the provided criteria.

    Examples
    --------
    >>> plot_p_infty_vs_time(df_from_dbs, ps=[0.6, 0.7], pd=0.1)
    """

    # Ensure ps is a list
    if isinstance(ps, float):
        ps = [ps]

    # Filter the DataFrame for the given pd and ps values
    filtered_df = mean_df.loc[(mean_df["pd"] == pd) & (mean_df["ps"].isin(ps))]

    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        raise ValueError("No data matches the provided criteria.")

    # Plotting
    plt.figure(figsize=(10, 6))

    cmap = cm.get_cmap("viridis", len(ps))

    for idx, ps_value in enumerate(ps):
        ps_df = filtered_df[filtered_df["ps"] == ps_value]
        # Ensure colors are evenly distributed in the colormap
        color = cmap(idx / (len(ps) - 1))

        plt.errorbar(
            ps_df["time"],
            ps_df["active_stem_cells_indicator"],
            yerr=ps_df["active_stem_cells_indicator_std"],
            label=f"$P_\infty (t)$ for $p_s=${ps_value}",
            color=color,
            uplims=True,
            lolims=True,
            marker=".",
            linestyle="-",
        )

    plt.xlabel("$t$")
    plt.ylabel("$P_\infty$")
    plt.title(f"$P_\infty (t)$ for $p_d = {pd}$")
    plt.legend()

    if log:
        plt.yscale("log")

    plt.grid(True)
    plt.show()


def p_infty_of_ps(p_s, p_c, c):
    """
    The function to fit to the data.

    Parameters
    ----------
    p_s : float or array-like
        The values of ps.
    p_c : float
        The critical percolation probability.
    c : float
        The constant.

    Returns
    -------
    result : float or array-like
        The computed values of the function for the given parameters.
    """
    return 0.5 * erf((p_s - p_c) / c) + 0.5


def plot_p_infty_vs_ps(
    mean_df: pd.DataFrame,
    time_steps: List[int],
    pd: float = 0.0,
    fit: bool = False,
    save: bool = False,
    path_to_save: str = None,
    dpi: int = 600,
) -> None:
    """
    Plot P_infty as a function of ps for different time steps and for each pd,
    optionally fitting a function to the points.

    Parameters
    ----------
    mean_df : pd.DataFrame
        DataFrame containing the data, in the form output by
        `average_over_realizations` (it should include the
        'active_stem_cells_indicator' column).
    time_steps : list of int
        The list of time steps to plot.
    pd_values : list of float, default=[0]
        The list of p_d values to consider.
    fit : bool, default=False
        Whether to fit a function to the points.
    save : bool, default=False
        Whether to save the plot to a file.
    path_to_save : str, optional
        The path to save the plot to. If not provided, defaults to a name based
        on the parameters.

    Raises
    ------
    ValueError
        If no data matches the provided criteria.

    Examples
    --------
    >>> plot_p_infty_vs_ps(
    >>>     mean_df,
    >>>     time_steps=[10, 20, 30],
    >>>     pd_values=0.1,
    >>>     fit=True,
    >>>     save=True,
    >>>     path_to_save='plot.png',
    >>> )
    """

    # Define color map for different time steps
    cmap = plt.cm.magma

    # Define figure and axis objects
    fig, ax = plt.subplots()

    # Loop over time steps and plot active_stem_cells_indicator as a function of ps
    for i, t in enumerate(time_steps):
        df_time = mean_df[(mean_df["time"] == t) & (mean_df["pd"] == pd)]
        if df_time.empty:
            continue

        if fit:
            # Plot the data points
            ax.scatter(
                df_time["ps"],
                df_time["active_stem_cells_indicator"],
                color=cmap(i / len(time_steps)),
                marker=".",
                label=f"Time: {t}",
            )

            # Perform the curve fit
            popt, _ = sp.optimize.curve_fit(
                p_infty_of_ps,
                df_time["ps"],
                df_time["active_stem_cells_indicator"],
                p0=(0.7, 0.1),
                maxfev=5000,
                bounds=((0, 0), (1, 1)),
            )

            # Generate x values for the fitted function
            x_values = np.linspace(
                min(df_time["ps"]), max(df_time["ps"]), num=100
            )

            # Generate y values for the fitted function
            y_values = p_infty_of_ps(x_values, *popt)

            # Plot the fitted function
            ax.plot(
                x_values,
                y_values,
                color=cmap(i / len(time_steps)),
                linestyle="--",
                marker=".",
                label=f"Fitted function: Time {t}",
            )
        else:
            ax.plot(
                df_time["ps"],
                df_time["active_stem_cells_indicator"],
                color=cmap(i / len(time_steps)),
                marker=".",
                linestyle="--",
                label=f"Time: {t}",
            )

    # Set axis labels and legend
    ax.set_xlabel("$p_s$", fontsize=14, color="black")
    ax.set_ylabel("$P_{\infty}$", fontsize=14, color="black")
    ax.legend(title="Time Steps", bbox_to_anchor=(0.05, 1), loc="upper left")

    if save:
        if path_to_save is None:
            path_to_save = f"p_infty_vs_ps_pd_{pd}_steps_{'_'.join(map(str, time_steps))}.png"
        plt.savefig(path_to_save, dpi=dpi)
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()


def plot_fitted_pc_vs_t(mean_df, output_path, pd_values=[0]):
    """Fits the function to the data for each time step and each pd, and plots
    the fitted percolation probability. Also, it generates a CSV file for each
    p_d value, storing 3 columns with the value for t, and the corresponding
    values of pc and c.

    Parameters
    ----------
    mean_df : pandas.DataFrame
        The DataFrame containing the data to fit the function to.
    pd_values : list of float
        The list of p_d values to consider.
    output_path : str
        The base path to save the plots and CSV files to. Each plot will be saved to a file with name
        constructed as {output_path}_pd_{pd_value}.png, and each CSV file as {output_path}_pd_{pd_value}.csv.
    """

    bnds = ((0, 0), (1, 1))  # bounds for the parameters

    for pd_value in pd_values:
        popt = []
        pcov = []
        list_of_pc = []
        list_of_c = []

        # Loop over time steps
        times_of_observation = sorted(set(mean_df["time"]))
        for t in times_of_observation:
            df_time = mean_df[
                (mean_df["time"] == t) & (mean_df["pd"] == pd_value)
            ]
            # we fit with scipy
            popt_i, pcov_i = curve_fit(
                p_infty_of_ps,
                df_time["ps"],
                df_time["active_stem_cells_indicator"],
                p0=(0.7, 0.1),
                maxfev=5000,
                bounds=bnds,
            )
            popt.append(popt_i)
            pcov.append(pcov_i)
            list_of_pc.append(popt_i[0])
            list_of_c.append(popt_i[1])

        # Save the parameters to a CSV file
        pd.DataFrame(
            {
                "t": times_of_observation,
                "pc": list_of_pc,
                "c": list_of_c,
            }
        ).to_csv(f"{output_path}_pd_{pd_value}.csv", index=False)

        fig, ax = plt.subplots()
        ax.plot(
            times_of_observation,
            list_of_pc,
            marker=".",
            linestyle="dashed",
            label=f"$p_c(t)$",
        )

        ax.set_xlabel("$t$", fontsize=12, color="black")
        ax.set_ylabel("$p_c$", fontsize=12, color="black")
        ax.set_title(
            f"Fitted percolation probability vs time for $p_d = {pd_value}$",
            fontdict={"fontsize": 12},
        )
        ax.legend()

        plt.savefig(f"{output_path}_pd_{pd_value}.png", dpi=600)
        plt.close(fig)  # Close the figure to free up memory


def create_heatmap(df, output_path, pd_values=[0]):
    """Creates and saves a heatmap of P_infty for each given probability pd.

    The function creates a heatmap for each given 'pd' value and saves it to
    the indicated working directory as a .png file. The file is named
    'heatmap_pd_{}.png', where {} is replaced by the 'pd' value. The heatmap's
    color intensity indicates the value of 'active_stem_cells_indicator'.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the simulation data. It must include 'pd',
        'ps', 'time', and 'active_stem_cells_indicator' columns.
    output_path : str
        The path for saving the plot.
    pd_values : list
        The list of unique probabilities pd values for which the heatmap
        should be created.

    Returns
    -------
    None

    Examples
    --------
    >>> output_path = "/path/to/save/"
    >>> create_heatmap(df, output_path, df['pd'].unique())

    """
    for pd in pd_values:
        # Filter the DataFrame for the current 'pd' value
        df_filtered = df[df["pd"] == pd]

        # Pivot the DataFrame
        df_pivot = df_filtered.pivot(
            index="ps", columns="time", values="active_stem_cells_indicator"
        )

        # Create a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_pivot, cmap="magma", annot=True, fmt=".2f")

        plt.title(
            f"Heatmap of active_stem_cells_indicator by $p_s$ and $t$ for $p_d$ = {pd}"
        )
        plt.xlabel("$t$")
        plt.ylabel("$p_s$")

        # Save the heatmap to a file
        plt.savefig(f"{output_path}heatmap_pd_{pd}.png", dpi=600)
        plt.close()


def create_pc_heatmap(mean_df, output_path, time_step):
    """Creates and saves a heatmap of pc for a given time step.

    Usually, we want to see the critical probability pc for the latest time
    possible. This may be limitted by the latest step performed by all
    simulations simultaneously. An alternative in the future could be to use
    the maximum value of a sigmoide function fitted to pc(t).

    Parameters
    ----------
    mean_df : pandas.DataFrame
        The DataFrame containing the computed means.
    output_path : str
        The path for saving the plot.
    time_step : int
        The time step for which to create the heatmap.

    Returns
    -------
    None
    """
    bnds = ((0, 0), (1, 1))  # bounds for the parameters

    df_filtered = mean_df[mean_df["time"] == time_step]
    pc_values = []

    # Loop over unique pd and ps values
    for p_d in df_filtered["pd"].unique():
        for ps in df_filtered["ps"].unique():
            # Filter for current pd and ps
            df_ps_pd = df_filtered[
                (df_filtered["pd"] == p_d) & (df_filtered["ps"] == ps)
            ]

            # Fit the function to the data
            popt, _ = curve_fit(
                p_infty_of_ps,
                df_ps_pd["ps"],
                df_ps_pd["active_stem_cells_indicator"],
                p0=(0.7, 0.1),
                maxfev=5000,
                bounds=bnds,
            )

            pc_values.append(
                {
                    "pd": p_d,
                    "ps": ps,
                    "pc": popt[0],  # popt[0] is the fitted value of pc
                }
            )

    # Create a DataFrame from the pc_values list of dictionaries
    df_pc = pd.DataFrame(pc_values)

    # Sort the DataFrame by pd and ps
    df_pc = df_pc.sort_values(by=["pd", "ps"])

    # Pivot the DataFrame to create a grid of ps, pd, and pc values
    df_pivot = df_pc.pivot(index="pd", columns="ps", values="pc")

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_pivot, cmap="magma", cbar_kws={"label": "$p_c$"})
    plt.title(f"Critic percolation probability $p_c$ at time {time_step}")
    plt.xlabel("$p_s$")
    plt.ylabel("$p_d$")

    # Invert the y-axis to have pd in ascending order from bottom to top
    plt.gca().invert_yaxis()

    # Save the heatmap to a file
    plt.savefig(f"{output_path}pc_heatmap_t_{time_step}.png", dpi=600)
    plt.close()


if __name__ == "__main__":
    # set to `False` if you are trying to process a `.csv` generated from the
    # old `.dat` files. Leave it as `True` for processing `.csv` DataFrames
    # generated from the standard `.db` merged database.
    db_files = True

    csv_file = "/home/nate/Devel/tumorsphere_culture/examples/playground/df_simulations.csv"
    plot1_output_path = "/home/nate/Devel/tumorsphere_culture/examples/playground/post_processed/p_infty_vs_ps"
    plot2_output_path = "/home/nate/Devel/tumorsphere_culture/examples/playground/post_processed/pc_vs_t"
    output_path = "/home/nate/Devel/tumorsphere_culture/examples/playground/post_processed/"

    set_plot_style()
    df = read_data(csv_file)
    if not db_files:
        # db files don't have this need
        df = add_zero_time_point(df)
    df = add_active_stem_cells_indicator(df)
    mean_df = average_over_realizations(df)
    plot_p_infty_vs_ps(
        mean_df, [4, 6, 8, 10], plot1_output_path, pd_values=[0]
    )
    plot_p_infty_vs_ps_with_fit(
        mean_df, [4, 6, 8, 10], plot1_output_path, pd_values=[0]
    )
    plot_fitted_pc_vs_t(mean_df, plot2_output_path, pd_values=[0])
    create_heatmap(mean_df, output_path, pd_values=[0])
    create_pc_heatmap(mean_df, output_path, time_step=10)
