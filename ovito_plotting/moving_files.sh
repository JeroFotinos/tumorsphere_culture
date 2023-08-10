#!/bin/bash

# This should be executed from the main folder.

# Deleting old files
rm ovito_plotting/no_huevo/realization_0/ovito_data_culture_pd\=0.0_ps\=0.7_rng_seed\=1052030010736714.*
rm ovito_plotting/no_huevo/realization_2/ovito_data_culture_pd\=0.0_ps\=0.7_rng_seed\=853827592514924.*
rm ovito_plotting/no_huevo/realization_1/ovito_data_culture_pd\=0.0_ps\=0.7_rng_seed\=1099062448631258.*

# Moving new files to the correct folder
mv ovito_data_culture_pd\=0.0_ps\=0.7_rng_seed\=1052030010736714.* ovito_plotting/no_huevo/realization_0/
mv ovito_data_culture_pd\=0.0_ps\=0.7_rng_seed\=1099062448631258.* ovito_plotting/no_huevo/realization_1/
mv ovito_data_culture_pd\=0.0_ps\=0.7_rng_seed\=853827592514924.* ovito_plotting/no_huevo/realization_2/
