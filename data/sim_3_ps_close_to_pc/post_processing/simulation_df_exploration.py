"""simulation_df_exploration.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mCjhxew8ZW9t85_cPk3XOumpdhXk0UXp

# Tumorsphere DataFrame Exploration
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

# set matplotlib style
plt.style.use("ggplot")
plt.rcParams["axes.edgecolor"] = "darkgray"
plt.rcParams["axes.linewidth"] = 0.8

# read csv generated by df_for_simulations.py
csv_file = "/home/nate/Devel/tumorsphere_culture/data/sim_3_ps_close_to_pc/p_infty_vs_ps_fit/df_simulations.csv"

with open(csv_file, "r") as file:
    df = pd.read_csv(file)

# Los datos empiezan en t = 1, pero les falta el t = 0 donde tenemos
# sólo una CSC. Agrego esto para todas las simulaciones

# Define the new rows as a dictionary
new_rows = {
    "ps": [],
    "n": [],
    "time": [],
    "total_cells": [],
    "active_cells": [],
    "stem_cells": [],
    "active_stem_cells": [],
}

# Loop over unique values of ps and n
for ps in df["ps"].unique():
    for n in df[df["ps"] == ps]["n"].unique():
        # Add the new row
        new_rows["ps"].append(ps)
        new_rows["n"].append(n)
        new_rows["time"].append(0)
        new_rows["total_cells"].append(1)
        new_rows["active_cells"].append(1)
        new_rows["stem_cells"].append(1)
        new_rows["active_stem_cells"].append(1)

# Append the new rows to the original DataFrame
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Son casi 6100 filas, que corresponden a 10 valores de $p_s$, por 10
# realizaciones para cada uno, con 61 pasos temporales cada una (contando
# el $t = 0$; algunas no llegaron a $t = 60$, por eso da menos).
# Quedó joya armado el `DataFrame`.

# Lo que quiero hacer ahora, es armar $P_\infty (p_s)$.


# To compute the booleans that indicate the presence of Active CSCs, I came up
# with two different methods. Let us verify if they match.

# Using lambda function
df["active_stem_cells_bool1"] = df["active_stem_cells"].apply(
    lambda x: 1 if x > 0 else 0
)

# Using numpy.sign
df["active_stem_cells_bool2"] = np.sign(df["active_stem_cells"]).astype(int)

# Compare the two columns
assert all(df["active_stem_cells_bool1"] == df["active_stem_cells_bool2"])

# It passes, so I can use either of those.

cols_to_average = [
    "total_cells",
    "active_cells",
    "stem_cells",
    "active_stem_cells",
    "active_stem_cells_bool2",
]

# Group by 'ps' and 'time' columns, and compute mean for the remaining columns
mean_df = df.groupby(["ps", "time"])[cols_to_average].mean().reset_index()


# Que tenga exactamente 610 filas me deja bastante tranquilo, ya que sí o sí
# tenemos los 10 valores de $p_s$, con 61 pasos en al menos alguna realización.
# No necesariamente a todo tiempo tenemos 10 realizaciones, pero estamos
# promediando sobre las realizaciones.

# Tratemos ahora de reproducir mi figura para $P_{\infty}(p_s)$.


# # Set style and context for seaborn
# sns.set_style("whitegrid")
# sns.set_context("notebook", font_scale=1.2)

# Define color map for different time steps
cmap = plt.cm.magma

# Define list of time steps to plot
list_of_steps = [55, 45, 35, 30, 25]

# Define figure and axis objects
fig, ax = plt.subplots()

# Loop over time steps and plot active_stem_cells_indicator as a function of ps
for i, t in enumerate(list_of_steps):
    df_time = mean_df[mean_df["time"] == t]
    ax.plot(
        df_time["ps"],
        df_time["active_stem_cells_bool2"],
        color=cmap(i / len(list_of_steps)),
        marker=".",
        linestyle="--",
        label=f"Time: {t}",
    )

# Set axis labels and legend
ax.set_xlabel("$p_s$", fontsize=14, color="black")
ax.set_ylabel("$P_{\infty}$", fontsize=14, color="black")
ax.legend(title="Time Steps", bbox_to_anchor=(0.05, 1), loc="upper left")

plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_3_ps_close_to_pc/p_infty_vs_ps_fit/p_infty_vs_ps.png",
    dpi=600,
)

# ¡Da igual! Ahora veamos $p_c(t)$.


# First, we define the function to fit
def p_infty_of_ps(p_s, p_c, c):
    return 0.5 * sp.special.erf((p_s - p_c) / c) + 0.5


# We do the fit for every step in the list of steps and save it to a file
popt = []
pcov = []
list_of_pc = []
list_of_c = []
bnds = ((0, 0), (1, 1))
# notation: ((lower_bound_1st_param, lower_bound_2nd_param),
# (upper_bound_1st_param, upper_bound_2nd_param))

# Loop over time steps and plot active_stem_cells_indicator as a function of ps
times_of_observation = list(set(mean_df["time"]))
for t in times_of_observation:
    df_time = mean_df[mean_df["time"] == t]
    # we fit with scipy
    popt_i, pcov_i = sp.optimize.curve_fit(
        p_infty_of_ps,
        df_time["ps"],
        df_time["active_stem_cells_bool2"],
        p0=(0.7, 0.1),
        maxfev=5000,
        bounds=bnds,
    )
    popt.append(popt_i)
    pcov.append(pcov_i)
    list_of_pc.append(popt_i[0])
    list_of_c.append(popt_i[1])

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
    "Fitted percolation probability vs time of observation",
    fontdict={"fontsize": 12},
)
ax.legend()

plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_3_ps_close_to_pc/p_infty_vs_ps_fit/pc_vs_t_df_version.png",
    dpi=600,
)


def create_heatmap(df, output_path):
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

    # Filter the DataFrame for the current 'pd' value
    df_filtered = df

    # Pivot the DataFrame
    df_pivot = df_filtered.pivot(
        index="ps", columns="time", values="active_stem_cells_bool2"
    )

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_pivot, cmap="magma", annot=False, fmt=".2f")

    plt.title("Heatmap of $P_\infty$ by $p_s$ and $t$ for $p_d$ = 0")
    plt.xlabel("$t$")
    plt.ylabel("$p_s$")

    # Save the heatmap to a file
    plt.savefig(f"{output_path}heatmap_pd_0.png", dpi=600)
    plt.close()


output_path = "/home/nate/Devel/tumorsphere_culture/data/sim_3_ps_close_to_pc/p_infty_vs_ps_fit/"
create_heatmap(mean_df, output_path)
