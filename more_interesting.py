from cells import *
from culture import *

csc_culture = Culture(first_cell_type='csc', prob_stem=0.5, cell_max_repro_attempts=1000)
csc_culture.simulate(7)
print(len(csc_culture.cells))
csc_culture.plot_culture_spheres()