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
    print_statistics: bool = False,
    plot_err_bars: bool = True,
    fit_with_y_err: bool = False,
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
    pd : float, default=0.0
        The p_d value to consider.
    fit : bool, default=False
        Whether to fit a function to the points.
    save : bool, default=False
        Whether to save the plot to a file.
    path_to_save : str, optional
        The path to save the plot to. If not provided, defaults to a name based
        on the parameters.
    dpi : int, default=600
        The resolution in dots per inch for the saved plot.
    print_statistics : bool, default=False
        Whether to print fitting statistics.
    plot_err_bars : bool, default=False
        Whether to plot error bars for the data points.
    fit_with_y_err : bool, default=False
        Whether to fit the function using y-errors (uncertainties in the data
        points).

    Raises
    ------
    ValueError
        If no data matches the provided criteria.

    Examples
    --------
    >>> plot_p_infty_vs_ps(
    >>>     mean_df,
    >>>     time_steps=[10, 20, 30],
    >>>     pd=0.1,
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

        yerr = (
            df_time["active_stem_cells_indicator_std"]
            if plot_err_bars
            else None
        )

        if fit:
            sigma = (
                df_time["active_stem_cells_indicator_std"]
                if fit_with_y_err
                else None
            )

            # Plot the data points with error bars if specified
            ax.errorbar(
                df_time["ps"],
                df_time["active_stem_cells_indicator"],
                yerr=yerr,
                color=cmap(i / len(time_steps)),
                marker=".",
                uplims=True,
                lolims=True,
                linestyle="None",
                label=f"Time: {t}",
            )

            # Perform the curve fit
            popt, pcov = sp.optimize.curve_fit(
                p_infty_of_ps,
                df_time["ps"],
                df_time["active_stem_cells_indicator"],
                p0=(0.7, 0.1),
                maxfev=5000,
                bounds=((0, 0), (1, 1)),
                sigma=sigma,
            )

            # Print the results of the fittings
            if print_statistics:
                print(f"----------- Time {t} -----------")
                print("Fitting parameters:")
                print(f"  p_c (critical percolation probability): {popt[0]}")
                print(f"  c (constant): {popt[1]}")
                print("Covariance of parameters (covariance matrix):")
                print(f"  {pcov}\n")
                # print('--------------------------------')

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
                label=f"Fitted function: Time {t}",
            )
        else:
            ax.errorbar(
                df_time["ps"],
                df_time["active_stem_cells_indicator"],
                yerr=yerr,
                color=cmap(i / len(time_steps)),
                marker=".",
                uplims=True,
                lolims=True,
                linestyle="--",
                label=f"Time: {t}",
            )

    # Set axis labels and legend
    ax.set_xlabel("$p_s$", fontsize=14, color="black")
    ax.set_ylabel("$P_{\infty}$", fontsize=14, color="black")
    ax.legend(title="Time Steps", bbox_to_anchor=(0.05, 1), loc="upper left")
    ax.grid(True)

    if save:
        if path_to_save is None:
            path_to_save = f"p_infty_vs_ps_pd_{pd}_steps_{'_'.join(map(str, time_steps))}.png"
        plt.savefig(path_to_save, dpi=dpi)
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()


def make_df_fitted_pc_vs_t(
    mean_df: pd.DataFrame, pd_value: float = 0.0, fit_with_uncert: bool = True
) -> pd.DataFrame:
    """
    Fits the function to the data for each time step and each pd, and returns
    a DataFrame with the fitted percolation probability and constant, along
    with their standard errors.

    It requires a DataFrame as output by `average_over_realizations`, with a
    column for the average of the 'active_stem_cells_indicator'.

    Parameters
    ----------
    mean_df : pandas.DataFrame
        The DataFrame containing the data to fit the function to.
    pd_value : float, default=0.0
        The p_d value to consider.
    fit_with_uncert : bool, default=True
        Whether to fit the data using the uncertainties in
        active_stem_cells_indicator_std.

    Returns
    -------
    fitted_df : pandas.DataFrame
        The DataFrame containing the time, fitted percolation probability
        $p_c$, constant $c$, and their standard errors.
    """
    bnds = ((0, 0), (1, 1))  # bounds for the parameters

    times_of_observation = sorted(set(mean_df["time"]))
    list_of_pc = []
    list_of_c = []
    list_of_pc_err = []
    list_of_c_err = []

    # Loop over time steps
    for t in times_of_observation:
        df_time = mean_df[(mean_df["time"] == t) & (mean_df["pd"] == pd_value)]
        if (
            fit_with_uncert
            and "active_stem_cells_indicator_std" in df_time.columns
        ):
            sigma = df_time["active_stem_cells_indicator_std"]
        else:
            sigma = None

        # we fit with scipy
        popt_i, pcov_i = curve_fit(
            p_infty_of_ps,
            df_time["ps"],
            df_time["active_stem_cells_indicator"],
            p0=(0.7, 0.1),
            maxfev=5000,
            bounds=bnds,
            sigma=sigma,
        )
        list_of_pc.append(popt_i[0])
        list_of_c.append(popt_i[1])
        list_of_pc_err.append(np.sqrt(pcov_i[0, 0]))  # Standard error of pc
        list_of_c_err.append(np.sqrt(pcov_i[1, 1]))  # Standard error of c

    fitted_df = pd.DataFrame(
        {
            "t": times_of_observation,
            "pc": list_of_pc,
            "pc_err": list_of_pc_err,
            "c": list_of_c,
            "c_err": list_of_c_err,
        }
    )

    return fitted_df


def plot_fitted_pc_vs_t(
    fitted_df: pd.DataFrame,
    output_path: str = "",
    pd_value: float = 0.0,
    save: bool = False,
    dpi: int = 600,
    plot_err_bars: bool = False,
) -> None:
    """
    Plots the fitted percolation probability as a function of time.

    It requires a DataFrame as output by `make_df_fitted_pc_vs_t`. If save is
    set to True, the plot is saved to a file named
    'fitted_pc_vs_t_pd_{pd}.png'.

    Parameters
    ----------
    fitted_df : pandas.DataFrame
        The DataFrame containing the time, fitted percolation probability
        $p_c$, constant $c$, and their standard errors.
    output_path : str
        The path to save the plot.
    pd_value : float, default=0.0
        The p_d value to consider.
    save : bool, default=True
        Whether to save the plot.
    dpi : int, default=600
        The resolution in dots per inch for the saved plot.
    plot_err_bars : bool, default=False
        Whether to plot error bars.

    Returns
    -------
    None
    """
    times_of_observation = fitted_df["t"]
    list_of_pc = fitted_df["pc"]
    list_of_pc_err = fitted_df["pc_err"] if plot_err_bars else None

    fig, ax = plt.subplots()
    ax.errorbar(
        times_of_observation,
        list_of_pc,
        yerr=list_of_pc_err,
        fmt="." if plot_err_bars else "o",
        uplims=True,
        lolims=True,
        linestyle="dashed",
        label="$p_c(t)$",
    )

    ax.set_xlabel("$t$", fontsize=12, color="black")
    ax.set_ylabel("$p_c$", fontsize=12, color="black")
    ax.set_title(
        f"Fitted critic percolation probability vs time for $p_d = {pd_value}$",
        fontdict={"fontsize": 12},
    )
    ax.legend()
    ax.grid(True)

    if save:
        plt.savefig(f"{output_path}fitted_pc_vs_t_pd_{pd_value}.png", dpi=dpi)
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()


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
