from tumorsphere.culture import Culture

# Create a culture object
cult = Culture(
    first_cell_is_stem=True,
    prob_stem=0.36,
    prob_diff=0,
    measure_time=True,
)

# Simulate culture growth
cult.simulate_with_persistent_data(
    num_times=60, culture_name="culture_pd=0_ps=0.36_realization_1"
)
