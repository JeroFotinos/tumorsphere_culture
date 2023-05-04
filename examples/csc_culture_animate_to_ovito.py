from tumorsphere.cells import *
from tumorsphere.culture import *

import pickle

csc_culture = Culture(
    first_cell_is_stem=True, prob_stem=0.75, cell_max_repro_attempts=1000
)

num_steps = 10

with open('csc_culture_0.pkl', 'wb') as file:
    pickle.dump(csc_culture, file)


for i in range(num_steps):
    csc_culture.simulate(1)
    with open(f'csc_culture_{i+1}.pkl', 'wb') as file:
        pickle.dump(csc_culture, file)

