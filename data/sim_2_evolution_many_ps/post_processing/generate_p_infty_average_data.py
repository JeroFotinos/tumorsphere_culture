import numpy as np

import glob

# List all the files with the pattern 'ps-n.dat'
file_list = glob.glob(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/dat_files/*.dat"
)

# Create a dictionary to store the data and the counts for each (p, i) pair
data_dict = {}
count_dict = {}

# Loop through each file
for file_name in file_list:
    # Get the p value from the file name
    p = float(file_name.split("/")[-1].split("-")[0])
    # we do a .splot("/") to eliminate the path from the name string

    # Read the data from the file
    with open(file_name, "r") as file:
        file_data = file.readlines()
        # file_data is a list of strings (one for each line of the file)

    # Loop through each line in the file
    for i, line in enumerate(file_data):
        # Convert the line to a list of integers, and take sign to convert
        # to 0 or 1 (either there are still cells in that category or not)
        line_data = [np.sign(float(x)) for x in line.strip().split(", ")]
        # = [total(i), active(i), total stem(i), active stem(i)]

        # Add the line data to the dictionary ('i' is the line number,
        # i.e. the time step, and 'line' is the corresponding line string)
        if (p, i) not in data_dict.keys():
            data_dict[(p, i)] = line_data
            count_dict[(p, i)] = 1
            # add the key and initialize counter
            # for this ps and time step values
        else:
            data_dict[(p, i)] = [
                sum(x) for x in zip(data_dict[(p, i)], line_data)
            ]
            count_dict[(p, i)] += 1
            # Explanation: let's say that
            # data_dict[(p, i)] = [a1, b1, c1, d1]
            # and that
            # line_data = [a2, b2, c2, d2]
            # then,
            # zip(data_dict[(p, i)], line_data) = ((a1, a2), (b1, b2), (c1, c2), (d1, d2))
            # and, finally, data_dict[(p, i)] gets replaced by
            # (sum(a1, a2), sum(b1, b2), sum(c1, c2), sum(d1, d2))
            # At the end of the day, data_dict[(p, i)] contains an array whose
            # j-th element is the sum over the realizations of the j-th element
            # of the i-th time step for p_s = p.
            #
            # Also, count_dict[(p, i)] tells us how many realizations do we have
            # for p_s = p and the time step i. If simulations are incomplete,
            # i.e. if we are working with preliminary data, we can still obtain
            # a correct average because we are taking into account the number
            # of realizations available.

# Loop through each unique ps value and save the time step and p_infty for that ps

for p in set([x[0] for x in data_dict.keys()]):
    # Create a new file for the averages
    with open(
        "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/p_infty_vs_t_averages/average-p_infty_vs_t-ps-{}.dat".format(
            p
        ),
        "w",
    ) as file:
        file.write(
            "{:.2f}, {:.4f}\n".format(
                0,  # time step
                1,  # active stem
            )
        )
        # Calculate the averages for this step
        for key in data_dict.keys():
            if key[0] == p:
                # Write the averages to the file
                file.write(
                    "{:.2f}, {:.4f}\n".format(
                        key[1] + 1,  # time step
                        # data_dict[key][0] / count_dict[key],  # total
                        # data_dict[key][1] / count_dict[key],  # active
                        # data_dict[key][2] / count_dict[key],  # total stem
                        data_dict[key][3] / count_dict[key],  # active stem
                    )
                )



# Loop through each unique ps value and save the ps and p_infty in a file that corresponds to the time step 

for i in set([(x[1] + 1) for x in data_dict.keys()]):
    # Create a new file for the averages
    with open(
        "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/p_infty_vs_ps_averages/average-p_infty_vs_ps-t-{}.dat".format(
            i
        ),
        "w",
    ) as file:
        # Calculate the averages for this step
        for key in data_dict.keys():
            if key[1] == i:
                # Write the averages to the file
                file.write(
                    "{:.2f}, {:.4f}\n".format(
                        key[0],  # ps
                        # data_dict[key][0] / count_dict[key],  # total
                        # data_dict[key][1] / count_dict[key],  # active
                        # data_dict[key][2] / count_dict[key],  # total stem
                        data_dict[key][3] / count_dict[key],  # active stem
                    )
                )