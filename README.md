# Nutritional Stability Metric Computation Repository

Nutritional stability (AKA nutristability) is a measurement of a the stability of a country's nutrient supply on a given year. This stability is measured by generating a bipartite network of the crops in the food system and the nutrients they contain, and performing an attack-tolerance process on that network. This approach into account the redundancy in sources of any particular nutrient.

This repository contains the code and data used to compute these values for the our study in Nature Communications, including scripts for early exploratory analysis.

## nutriStability-timeSeries.py

The central essential script for computing nutritional stability. 

### Inputs
Takes country, source (as defined in paper, P or PI), start year, end year, as command line arguments.

### Outputs
Returns a dataframe with a row for each year containing:
- year, 
- mean stability, 
- upper bound stability,
- lower bound stability,
- crop diversity (number of crops),
- the 5 most connected crops in the network
- the 5 least connected nutrients in the network
