'''
nutriStability-stabilitySweep -- Ben Emery 2019

Computes nutriStability at a range of possible link-weight thresholds for 
a country specified as an input argument, using only production data from 2016.

TO RUN FROM TERMINAL: python nutriStability-stabilitySweep.py $country
    $country is the country name 

    example run:
        python nutriStability-stabilitySweep.py Liberia
'''
 
# We at least need most of these packages.
import pandas as pd
import glob
import csv
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import pickle
import random
import collections
import sys

# Specify country, year, and threshold values to use
country = str(sys.argv[1])
year='2016'
threshes = np.arange(0.01,.70+0.01,.01)

# Import necessary data
production = pd.read_csv('data/Production_Crops_E_All_Data.csv',encoding = "ISO-8859-1")
foodNutrients = pd.read_csv('data/foodNutrients-frac.csv')
servingSizes = pickle.load(open('data/servingSizes.pkl','rb'))
population = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_63973.csv')

def multicurve_unweighted(H,lenL,numRuns):
    '''
    Draws the curve (and confidence bounds) given the bipartite network.
    '''
    df = pd.DataFrame()
    for i in range(numRuns):
        G = H.copy()
        height = []
        height.append(len(list(set([v for u,v in list(G.edges())]))))
        order = list(G.nodes())[:lenL]
        random.shuffle(order)
        for node in order:
            G.remove_node(node)
            height.append(len(list(set([v for u,v in list(G.edges())]))))
        df[i] = height
    df['avg'] = df.mean(axis=1)
    
    #there are 17 nutrients to begin with
    return((df['avg']/17).tolist())

def countryStability(countryName):
    '''
    This is the main function here. It calls the two functions above to compute everything.
    '''
    
    # Get lists of unique crops and nutrieents in dataset for the given country/year.
    countryCrops = [x for x in list(set(production.loc[(production['Area']==countryName)&(production['Y'+str(year)]>0),'Item'].tolist())) if x in list(servingSizes.keys())]
    nutrientList = [x for x in list(foodNutrients)[4:-1] if x!='Sodium']

    # Compute a stability for each threshold
    stabilities = []
    for thresh in threshes:
        # Initialize network from crops and nutrients
        bnk = nx.OrderedGraph()
        bnk.add_nodes_from(countryCrops, bipartite=0)
        bnk.add_nodes_from(nutrientList, bipartite=0)
        edges = []
        weights = []

        # add edges, determine weight, then remove edges that are too light
        for crop in countryCrops:
            newEdges = []
            for nutrient in nutrientList:
                weight = np.mean(foodNutrients.loc[foodNutrients['FAO_name']==crop,nutrient])
                weight = weight * 10**6 * 10**(-2) * (1/servingSizes[crop]) / population[population['Country Name']==countryName][year].iloc[0]
                weight = weight * np.mean(production.loc[(production['Area']==countryName)&(production['Item']==crop)]['Y'+year])
                if weight>thresh: # only add the edge if it's heavy enough
                    newEdges.append((crop, nutrient, weight))
                    weights.append(weight)
                edges.extend(newEdges)
                bnk.add_weighted_edges_from(newEdges) # put the edges on the network
            if bnk.degree[crop] == 0:
                bnk.remove_node(crop) # remove the crop if it had no edges connecting to it
                countryCrops = [x for x in countryCrops if x!=crop]
        
        # draw curves and take integrals
        curve = multicurve_unweighted(bnk,len(countryCrops),1000)    
        stabilities.append(sum([curve[i]/len(curve) for i in range(len(curve))]))
    return stabilities

# put all the data into an organized frame and save.
stabilities = countryStability(country)
df = pd.DataFrame()
df[country] = stabilities
df.to_csv('data/stabilities/'+country+'-stability.csv')





































