'''
nutriStability-timeseries -- Ben Emery 2019

Computes yearly nutriStability given a country and source (P, PI, or PIE)
Output includes yearly:
    - Nutristability, as well as upper and lower confidence bounds.
    - Average degree for crops, nutrients
        (This is the average number of nutrients "contained" by crops, and the average number foods that "contain" each nutrient)
    - Top 5 most connected (nutrient-diverse) crops
    - Bottom 5 LEAST connected (susceptible to loss) nutrients
    - Nutristability (mean and bounds) given the hypothetical nonexistence of each crop

TO RUN FROM TERMINAL: python nutriStability-timeSeries.py $country $source
    $country is the country name 
    $source is P (production), PI (production+imports), or PIE (production+imports-exports)

    example run:
        python nutriStability-timeseries.py Liberia PI
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
from scipy import stats

# Input arguments
country = str(sys.argv[1])
typ = str(sys.argv[2])

years = range(1961,2016+1)
thresh = 0.1

# Source type determines which dataset we pull from
if typ == 'P':
    production = pd.read_csv('data/production.csv',encoding = "ISO-8859-1")
elif typ == 'PI':
    production = pd.read_csv('data/prodPlusImp.csv',encoding = "ISO-8859-1")
elif typ == 'PIE':
    production = pd.read_csv('data/prodPlusImpMinusExp.csv',encoding = "ISO-8859-1")
else:
    print('Invalid type argument. Enter "P","PI", or "PIE".')

# Import the other data we need.
foodNutrients = pd.read_csv('data/foodNutrients-frac.csv')
servingSizes = pickle.load(open('data/servingSizes.pkl','rb'))
population = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_63973.csv')


def mean_confidence_interval(df, confidence=0.5):
    '''
    Computes the upper and lower confidence bounds for the curve's height at a certain point.
    '''
    n = len(list(df))
    m, se = df.mean(axis=1), df.sem(axis=1)
    h = stats.t.isf((1-confidence)/2,n)*se/np.sqrt(n)
    return m, m-h, m+h

def multicurve_unweighted(H,lenL,numRuns):
    '''
    Draws the curve given the bipartite network.
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
    df['avg'],df['lower'],df['upper'] = mean_confidence_interval(df)
    df = df.clip(0,17)
    
    #there are 17 nutrients to begin with
    return((df['avg']/17).tolist(),(df['lower']/17).tolist(),(df['upper']/17).tolist())

def countryStability(countryName):
    '''
    This is the main function here. It calls the two functions above to compute everything.
    '''
    nutrientList = [x for x in list(foodNutrients)[4:-1] if x!='Sodium']


    #OUTPUT LISTS (Each list is the length of the number of years)
    stabilities_m = [] # nutristability computed with curve
    stabilities_u = [] # nutristability computed with upper confidence band
    stabilities_l = [] # nutristability computed with lower confidence band
    numCrops = [] # number of crop nodes in network
    topCrops = [] # top five most connected crops
    avgDegCrops = [] # average degree of crop nodes 
    avgDegNutrs = [] # average degree of nutrient nodes
    botNutrs = [] #bottom five LEAST connected nutrients
    removed_crop_stabilities_list = [] #dictionary of nutristability (with bounds) for the network if each crop were not there
    

    for year in years:
        countryCrops = [x for x in list(set(production.loc[(production['Area']==countryName)&(production['Y'+str(year)]>0),'Item'].tolist())) if x in list(servingSizes.keys())]
        # Initialize network from crops and nutrients
        bnk = nx.OrderedGraph()
        bnk.add_nodes_from(countryCrops, bipartite=0)
        bnk.add_nodes_from(nutrientList, bipartite=0)
        edges = []
        weights = []
        cropDegrees = {}
        
        #add edges, determine weight, then remove edges that are too light
        for crop in countryCrops:
            newEdges = []
            for nutrient in nutrientList:
                weight = np.mean(foodNutrients.loc[foodNutrients['FAO_name']==crop,nutrient])
                weight = weight * 10**6 * 10**(-2) * (1/servingSizes[crop]) / population[population['Country Name']==countryName][str(year)].iloc[0]
                weight = weight * np.mean(production.loc[(production['Area']==countryName)&(production['Item']==crop)]['Y'+str(year)])
                if weight>0.1:
                    newEdges.append((crop, nutrient, weight))
                    weights.append(weight)
            edges.extend(newEdges)
            bnk.add_weighted_edges_from(newEdges)
            cropDegrees[crop] = bnk.degree[crop]
            if bnk.degree[crop] == 0:
                bnk.remove_node(crop)
                countryCrops = [x for x in countryCrops if x!=crop]
        nutrDegrees = {}
        for nutrient in nutrientList:
            nutrDegrees[nutrient] = bnk.degree[nutrient]


        # record network metrics
        botNutrs.append(sorted(nutrDegrees)[:5])
        topCrops.append(sorted(cropDegrees)[-5:])
        numCrops.append(len(countryCrops))
        avgDegCrops.append(np.mean([cropDegrees[x] for x in countryCrops]))
        avgDegNutrs.append(np.mean(list(nutrDegrees.values())))


        # draw curves and take integrals
        curve,lower,upper = multicurve_unweighted(bnk,len(countryCrops),100)    
        stabilities_m.append(sum([curve[i]/len(curve) for i in range(len(curve))]))
        stabilities_u.append(sum([upper[i]/len(upper) for i in range(len(upper))]))
        stabilities_l.append(sum([lower[i]/len(lower) for i in range(len(lower))]))


        # repeat with the removal of each crop to begin with
        removed_crop_stabilities={}
        for crop in countryCrops:
            bnkt = bnk.copy()
            bnkt.remove_node(crop)
            curve,lower,upper = multicurve_unweighted(bnkt,len(countryCrops),100)
            s_m=(sum([curve[i]/len(curve) for i in range(len(curve))]))
            s_u=(sum([upper[i]/len(upper) for i in range(len(upper))]))
            s_l=(sum([lower[i]/len(lower) for i in range(len(lower))]))
            removed_crop_stabilities[crop] = [s_m,s_l,s_u]
        removed_crop_stabilities_list.append(removed_crop_stabilities)




    return stabilities_m,stabilities_u,stabilities_l,numCrops,topCrops,botNutrs,avgDegCrops,avgDegNutrs,removed_crop_stabilities_list

# put all the data into an organized frame  and save.
stabilities_m,stabilities_u,stabilities_l,numCrops,topCrops,botNutrs,avgDegCrops,avgDegNutrs,removed_crop_stabilities_list = countryStability(country)
df = pd.DataFrame()
df[country+'_m'] = stabilities_m
df[country+'_l'] = stabilities_l
df[country+'_u'] = stabilities_u
df[country+'_numCrops'] = numCrops
df[country+'_topCrops'] = topCrops
df[country+'_botNutrs'] = botNutrs
df[country+'_avgDegCrops'] = avgDegCrops
df[country+'_avgDegNutrs'] = avgDegNutrs
df[country+'_removed_crop_stabilities'] = removed_crop_stabilities_list
df.to_csv('data/timeseries/nutriStability-timeseries-'+country+'-'+typ+'.csv')




