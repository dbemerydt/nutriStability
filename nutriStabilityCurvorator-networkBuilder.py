'''
nutriStabilityCurvorator -- Ben Emery 2019

Compute nutriStability and produce curve visual and a network visual given 
a country, year, link threshold, source (P or PI), plots the average curve.

TO RUN FROM TERMINAL: python nutriStabilityCurvorator.py $country $year $source
    $country is the country name 
    $year is year
    $source is P (production) PI (production+imports)

    example run:
        python nutriStabilityCurvorator.py Liberia 2015 PI
'''

import pandas as pd
import glob
import csv
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import sys
import random
import collections
import pickle
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
        df[i] = height # put list of heights into dataframe
    df['avg'] = df.mean(axis=1)
    
    #there are 17 nutrients to begin with
    return((df['avg']/17).tolist()) # normalize curves and export them as lists


countryName = sys.argv[1]
year = sys.argv[2]
typ = sys.argv[3]

# Source type determines which dataset we pull from
if typ == 'P':
    production = pd.read_csv('data/production.csv',encoding = "ISO-8859-1")
elif typ == 'PI':
    production = pd.read_csv('data/prodPlusImp.csv',encoding = "ISO-8859-1")
else:
    print('Invalid type argument -- enter "P" or "PI".')

# Import the other data we need.
foodNutrients = pd.read_csv('data/foodNutrients-frac.csv')
servingSizes = pickle.load(open('data/servingSizes.pkl','rb'))
population = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_63973.csv')

# Get lists of unique crops and nutrients in dataset for the given country/year.
countryCrops = [x for x in list(set(production.loc[(production['Area']==countryName)&(production['Y'+str(year)]>0),'Item'].to_list())) if x in list(servingSizes.keys())]
nutrientList = [x for x in list(foodNutrients)[4:-1] if x!='Sodium']

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
        if weight>0.1: # only add the edge if it's heavy enough
            newEdges.append((crop, nutrient, weight))
            weights.append(weight)
        edges.extend(newEdges)
        bnk.add_weighted_edges_from(newEdges) # put the edges on the network
    if bnk.degree[crop] == 0:
        bnk.remove_node(crop) # remove the crop if it had no edges connecting to it
        countryCrops = [x for x in countryCrops if x!=crop]


# draw curve and take integral
curve = multicurve_unweighted(bnk,len(countryCrops),1000)

# plot curve with nutriStability value included.
plt.plot([i/(len(curve)-1) for i in range(len(curve))],curve,color='dodgerblue')

ax = plt.gca()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.fill_between([i/(len(curve)-1) for i in range(len(curve))],0,curve,color='dodgerblue',alpha=0.5)
ax.fill_between([i/(len(curve)-1) for i in range(len(curve))],1,curve,color='red',alpha=0.5)
ax.set_aspect('equal','box')
plt.ylabel('proportion nutrients present')
plt.xlabel('proportion foods removed')

nS = sum([curve[i]/len(curve) for i in range(len(curve))])
plt.text(.5,.5,'{:.2}'.format(nS), color='red', ha='center', va='center', fontsize=50)
plt.title('nutriStability Curve: '+countryName+', '+year)
plt.savefig(countryName.lower()+'-'+year+'_nutriStabilityCurve.pdf',transparent=True,bbox_inches='tight')


plt.close()


# plot crop-nutrient bipartite network associated with country-year.
fig, ax = plt.subplots(figsize=(10,.4*len(countryCrops)))

pos=nx.drawing.layout.bipartite_layout(bnk,countryCrops)
# nx.draw_networkx(bnk,pos,with_labels=False,edge_color='green',width=[100*np.log10(w+1) for w in weights])
nx.draw_networkx(bnk,pos,with_labels=False,edge_color='green',width=[np.sqrt(w) for w in weights])
for crop in countryCrops:
    plt.text(pos[crop][0]-.05,pos[crop][1],s=crop,color='k',fontsize=20,ha='right')

for nutrient in nutrientList:
    plt.text(pos[nutrient][0]+.05,pos[nutrient][1],s=nutrient,color='k',fontsize=20,ha='left')



plt.axis('off')

plt.title('Crop-Nutrient Network: '+countryName,fontsize=30,y=.96)
# plt.savefig(countryName.lower()+'_bipartite-weighted.png',dpi=600,transparent=False,bbox_inches='tight')
plt.savefig(countryName.lower()+'_bipartite-weighted-'+year+'.pdf',transparent=True,bbox_inches='tight')

plt.close()
















