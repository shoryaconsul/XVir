#!/bin/bash

# This script lists possible commands to run XVir.
# MODELPATH="logs/experiment/XVir_models/XVir-2023.07.13-01-19-21/XVir-2023.07.13-01-19-21_2023.07.13-03-37-18.pt"
# MODELPATH="logs/experiment/XVir_models/XVir-2023.10.14-10-07-38/XVir-2023.10.14-10-07-38_2023.10.14-12-26-17.pt"
# MODELPATH="logs/experiment/XVir_models/XVir-2023.10.21-13-40-26/XVir-2023.10.21-13-40-26_2023.10.21-16-07-35.pt"

# MODELPATH="logs/experiment/XVir_models/XVir-2023.10.27-01-07-18/XVir-2023.10.27-01-07-18_2023.10.27-03-52-05.pt"
# MODELPATH="logs/experiment/XVir_models/XVir-2023.10.27-01-07-46/XVir-2023.10.27-01-07-46_2023.10.27-03-57-29.pt"
MODELPATH="logs/experiment/XVir_models/XVir-2023.10.27-01-08-29/XVir-2023.10.27-01-08-29_2023.10.27-04-07-36.pt"
# MODELPATH="logs/experiment/XVir_models/XVir-2023.10.27-01-09-38/XVir-2023.10.27-01-09-38_2023.10.27-04-10-54.pt"

DATAPATH="data/reads"
DATAFILE="split/test_data_150bp.pkl"
# DATAFILE="mutated_data_150bp.pkl"
# DATAFILE="mut_15_data_150bp.pkl"

python main.py --eval-only --model-path $MODELPATH --data-path $DATAPATH --data-file $DATAFILE 