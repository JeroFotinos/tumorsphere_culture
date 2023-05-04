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
# list_of_steps = np.arange(55, 24, -10)
list_of_steps = [59, 55, 45, 35, 30, 25]

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

# print(data_files)

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
bnds = ((0, -1e3), (1, 1e3))
# notation: ((lower_bound_1st_param, lower_bound_2nd_param), (upper_bound_1st_param, upper_bound_2nd_param))

with open(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/p_infty_vs_ps_fit/fit_p_infty_vs_ps_output.txt",
    "w",
) as fit_result:
    fit_result.write("----------- Fitting Results -----------\n")

for step_index in range(len(list_of_steps)):
    popt_i, pcov_i = curve_fit(
        p_infty_of_ps,
        ps[step_index],
        p_infty[step_index],
        p0=(0.7, 1),
        maxfev=5000,
        bounds=bnds,
    )
    popt.append(popt_i)
    pcov.append(pcov_i)
    list_of_pc.append(popt_i[0])
    list_of_c.append(popt_i[1])
    with open(
        "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/p_infty_vs_ps_fit/fit_p_infty_vs_ps_output.txt",
        "a",
    ) as fit_result:
        fit_result.write(f"Fit results for t = {list_of_steps[step_index]} \n")
        fit_result.write(f"p_c = {popt_i[0]}, c = {popt_i[1]}\n")
        fit_result.write(f"pcov = \n{pcov_i}\n\n")


# THIS DOESN'T MAKE ANY SENSE, we expect p_c to grow!
# # we calculate the mean and std of the fitted p_c
# mean_pc = np.mean(list_of_pc)
# std_pc = np.std(list_of_pc)
# mean_c = np.mean(list_of_c)
# std_c = np.std(list_of_c)

# # we write it to the end of the previous file
# with open('./fit_p_infty_vs_ps_output.txt', 'a') as fit_result:
#     fit_result.write('\n------------------------\n')
#     fit_result.write(f'Mean and std of p_c over t : \n {mean_pc} \pm {std_pc} \n')
#     fit_result.write(f'Mean and std of c over t : \n {mean_c} \pm {std_c} \n')


# Plot the curves
fig, ax = plt.subplots()

xarg = np.linspace(0, 1, 10000)
for step_index in range(len(list_of_steps)):
    # we plot the data
    ax.plot(
        ps[step_index],
        p_infty[step_index],
        marker=".",
        linestyle="None",
        label=f"$t = {list_of_steps[step_index]}$",
        color=plt.cm.magma(step_index / len(list_of_steps)),
    )
    # we plot the fitted function
    ax.plot(
        xarg,
        p_infty_of_ps(xarg, popt[step_index][0], popt[step_index][1]),
        linestyle="dashed",
        color=plt.cm.magma(step_index / len(list_of_steps)),
    )

# we set the grid
# plt.grid(color="gray", linestyle="--", linewidth=0.5)

# Set y-axis scale to logarithmic
# plt.yscale("log")

ax.set_xlabel("$p_s$")
ax.set_ylabel("$P_\infty (p_s)$")
ax.set_title(
    "Probability of active CSC presence vs Probability of self-replication",
    fontdict={"fontsize": 12},
)
ax.legend()

# to see the figure
# plt.show()

# This make the x and y labels smaller and closer to the axes
# plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/p_infty_vs_ps_fit/fit-p_infty_vs_ps.png",
    dpi=600,
)
# the dpi (dots per inch) is set to 100 by default, but it's too low for me
