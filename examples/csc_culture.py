from tumorsphere.cells import *
from tumorsphere.culture import *

csc_culture = Culture(
    first_cell_is_stem=True, prob_stem=0.5, cell_max_repro_attempts=1000
)
csc_culture.simulate(7)
print(len(csc_culture.cells))
csc_culture.plot_culture_spheres()
