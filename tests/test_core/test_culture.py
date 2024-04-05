from tumorsphere.core.simulation import Simulation

def test_culture_instantiation_by_simulation():
    """Test that a Culture is correctly instantiated by a Simulation."""
    sim = Simulation(
        num_of_realizations=1,
        num_of_steps_per_realization=5,
    )
    sim.simulate_single_culture(sql=False)

    assert sim.cultures is not None
    # first we assert that the cultures dictionary is not empty
    assert sim.cultures != {}
    # then, we assert that the culture object is correctly instantiated
    culture = next(iter(sim.cultures.values()))
    assert culture.grid.torus is True
    assert culture.grid.bounds is None
    assert culture.grid.cube_size == 2
    assert culture.grid.hash_table != {}