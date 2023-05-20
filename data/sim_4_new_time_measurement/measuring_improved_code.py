from tumorsphere.culture import Culture
import re # for regular expressions

# # Create a culture object and simulate it
# cult = Culture(
#     first_cell_is_stem=True,
#     prob_stem=0.36,
#     prob_diff=0,
#     measure_time=True,
# )

# cult.simulate_with_persistent_data(
#     num_times=20, culture_name="culture_pd=0_ps=0.36_realization_1"
# )

# plot the data for the time measurement with the one in sim_1_time_measurement
with open("time_per_step.dat", "r") as file:
    time_list = file.readlines()
    # step_comma_time_list is a list of strings (one for each line of the file)
    # so we have to convert it to a list of integers for the first column
    # and a list of floats for the second column

list_of_steps_new = [float(re.findall(r"\d+\.\d+", time_list[i])[0]) for i in range(len(time_list))]
# In this script, re.findall(r"\d+\.\d+", time_string) returns a list of all
# parts of time_string that match the regular expression \d+\.\d+. This
# regular expression matches one or more digits (\d+), followed by a decimal
# point (\.), followed by one or more digits (\d+), essentially representing
# a floating point number.
# Since we only expect one number in the string, we can just take the first
# element of the list [0] and convert it to a float with float().

with open(
    "/home/nate/Devel/tumorsphere_culture/data/sim_1_time_measurement/dat_files/filename.txt",
    "r",
) as file:
    old_time_list = file.readlines()
    # step_comma_time_list is a list of strings (one for each line of the file)
    # so we have to convert it to a list of integers for the first column
    # and a list of floats for the second column

list_of_steps_old = [float(re.findall(r"\d+\.\d+", old_time_list[i])[0]) for i in range(len(old_time_list))]

# plot the data for the time measurement with the one in sim_1_time_measurement
import matplotlib.pyplot as plt

# we want y log scale
plt.yscale("log")

plt.plot(list_of_steps_new, marker=".", label="new")
plt.plot(list_of_steps_old, marker=".", label="old")

# we set the x range limit to 20
plt.xlim(0, 18)
#plt.ylim(0, 7000)

# we want the x ticks to be integers
plt.xticks(range(0, 19, 2))

# we set the grid
plt.grid(color="green", linestyle="--", linewidth=0.5)

# Add labels and title
xlabel = r"$n$"
ylabel = r"$t(n)$ [seconds]"
title = r"Time $t(n)$ to compute the $n$-th step"
# Here, the r before the title string indicates that it is a raw string,
# which allows us to use backslashes to specify LaTeX commands.

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)

# we want the legend in the top left corner
plt.legend(loc="upper left")

# plt.legend()
# plt.show()

# we save the figure
plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_4_new_time_measurement/new_vs_old_time_by_step.png",
    dpi=600,
)

# Note: this was a quick script and I didn't bother to make it general, so
# it looks for the file "time_per_step.dat" in the current directory, but when
# you ran it in VS Code, that is the directory tumorsphere/. Also, 
# culture_pd=0_ps=0.36_realization_1 gets saved in the directory
# tumorsphere/data/.