import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

# set matplotlib style
plt.style.use("ggplot")
plt.rcParams["axes.edgecolor"] = "darkgray"
plt.rcParams["axes.linewidth"] = 0.8

# Set the path to the directory containing the data files
data_dir = "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/p_infty_vs_ps_averages/"

# Set of values for p for which are available to plot
# list_of_steps = list(np.arange(59, 1, -1)) # don't do this! Provokes error
# list_of_steps = [i for i in range(1, 60)] # this also raises error
# list_of_steps = [59, 55, 45, 35, 30, 25] # it only likes this method
# Let's emulate it
list_of_steps = []
for i in range(1, 57):
    list_of_steps.append(i)


# Find the data files for the specified values of p
data_files = []
for step_index in range(len(list_of_steps)):
    files_for_i = []
    for file_name in os.listdir(data_dir):
        if file_name.startswith(
            "average-p_infty_vs_ps-t-{}".format(list_of_steps[step_index])
        ):
            files_for_i.append(file_name)
    files_for_i.sort()  # sort the file names for this p value
    for file_name in files_for_i:
        data_files.append(os.path.join(data_dir, file_name))

# Read the data from the files
data = []
for file_index in range(len(list_of_steps)):
    data.append(np.loadtxt(data_files[file_index], delimiter=",", skiprows=0))
# skiprows = 1 es para el caso en el que las columnas del csv tienen título

# Extract the columns of interest
ps = []
p_infty = []

# we order the arrays for growing ps
for step_index in range(len(list_of_steps)):
    indices_to_sort_array = np.array(np.argsort(data[step_index][:, 0]))
    ps.append(data[step_index][indices_to_sort_array, 0])
    p_infty.append(data[step_index][indices_to_sort_array, 1])


# ----------- Fit the data -----------
# First, we define the function to fit
def p_infty_of_ps(p_s, p_c, c):
    return 0.5 * erf((p_s - p_c) / c) + 0.5


# We do the fit for every step in the list of steps and save it to a file
popt = []
pcov = []
list_of_pc = []
list_of_c = []
# bnds = ((0, -1e3), (1, 1e3)) # no tengo idea de cuánto puede valer c
bnds = ((0, 0), (1, 1))  # los valores típicos para c están entre
# c = 0.167 para t = 25 y c = 0.021 para t = 59.
# notation: ((lower_bound_1st_param, lower_bound_2nd_param),
# (upper_bound_1st_param, upper_bound_2nd_param))

for step_index in range(len(list_of_steps)):
    popt_i, pcov_i = curve_fit(
        p_infty_of_ps,
        ps[step_index],
        p_infty[step_index],
        p0=(0.7, 0.1),
        maxfev=5000,
        bounds=bnds,
    )
    popt.append(popt_i)
    pcov.append(pcov_i)
    list_of_pc.append(popt_i[0])
    list_of_c.append(popt_i[1])


# ----------- Write data to a file -----------
with open(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/p_infty_vs_ps_fit/pc_vs_t.txt",
    "w",
) as data_pc_t:
    data_pc_t.write(f"t, p_c \n")
    for i in range(len(list_of_steps)):
        data_pc_t.write(f"{list_of_steps[i]}, {list_of_pc[i]}\n")

# let's see who is ruinning everything
print("Times (>20) for which p_c < 0.2: \n")
for i in range(20, len(list_of_steps)):
    if list_of_pc[i] < 0.2:
        print(f"t = {list_of_steps[i]}\n")
# out:
# t = 23
# t = 34
# t = 45
# t = 56
# Diferencia de 11! ... but why?
# De hecho, es peor todavía porque todos tienen el mismo valor, y coincide
# con el de t = 1 y 11, es p_c = 0.035008016959456774

# ----------- Plot the curves -----------
fig, ax = plt.subplots()
ax.plot(
    list_of_steps,
    list_of_pc,
    marker=".",
    linestyle="dashed",
    label=f"$p_c(t)$",
)

ax.set_xlabel("$t$")
ax.set_ylabel("$p_c$")
ax.set_title(
    "Fitted percolation probability vs time of observation",
    fontdict={"fontsize": 12},
)
ax.legend()

# to see the figure
# plt.show()

# This make the x and y labels smaller and closer to the axes
# plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/p_infty_vs_ps_fit/p_c_vs_t.png",
    dpi=600,
)
# the dpi (dots per inch) is set to 100 by default, but it's too low for me
