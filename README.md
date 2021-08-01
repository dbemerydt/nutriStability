# Nutritional Stability Metric Computation Repository

Nutritional stability (AKA nutristability) is a measurement of a the stability of a country's nutrient supply on a given year. This stability is measured by generating a bipartite network of the crops in the food system and the nutrients they contain, and performing an attack-tolerance process on that network. This approach into account the redundancy in sources of any particular nutrient.

This repository contains the code and data used to compute these values for the our study in Nature Communications (DOI pending), including scripts for early exploratory analysis.

## Abstract:

Nutritional stability – a food system’s capacity to provide sufficient nutrients despite disturbance – is an important, yet challenging to measure outcome of diversified agriculture. Using 55 years of data across 184 countries, we assemble 22,000 bipartite crop-nutrient networks to quantify nutritional stability by simulating crop and nutrient loss in a country, and assess its relationship to crop diversity across regions, over time and between imports versus in country production. We find a positive, saturating relationship between crop diversity and nutritional stability across countries, but also show that over time nutritional stability remained stagnant or decreased in all regions except Asia. These results are attributable to diminishing returns on crop diversity, with recent gains in crop diversity among crops with fewer nutrients, or with nutrients already in a country’s food system. Finally, imports are positively associated with crop diversity and nutritional stability, indicating that many countries’ nutritional stability is market exposed.

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



## nutriStabilityCurvorator.py

A script for plotting the nutriStability curve for a particular country/year.

### Inputs
Takes country, year, and source as command line arguments.

### Outputs
Returns a plot as a pdf file with the curve and the nutristability value.


## nutriStabilityCurvorator-networkBuilder.py 

For plotting the nutriStability curve and the crop-nutrient bipartite network for a country/year.

### Inputs
Takes country, year, and source as command line arguments.

### Outputs
Returns a plot as a pdf file with the curve and the nutristability value, and a plot of the crop-nutrient bipartite network.


