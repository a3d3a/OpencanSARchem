#!/bin/sh

# To run this script from the terminal, use the following command:
# nohup ./OpencanSARchem.sh &

jupyter nbconvert --to notebook --execute OpencanSARchem.ipynb --inplace
