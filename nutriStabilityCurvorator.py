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


countryName = sys.argv[1]
year = sys.argv[2]
thresh = sys.argv[3]

production = pd.read_csv('data/Production_Crops_E_All_Data.csv',encoding = "ISO-8859-1")
foodNutrients = pd.read_csv('data/foodNutrients-frac.csv')
servingSizes = pickle.load(open('data/servingSizes.pkl','rb'))
population = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_63973.csv')


countryCrops = [x for x in list(set(production.loc[(production['Area']==countryName)&(production['Y'+str(year)]>0),'Item'].to_list())) if x in list(servingSizes.keys())]
nutrientList = list(foodNutrients)[4:-1]

bnk = nx.OrderedGraph()
bnk.add_nodes_from(countryCrops, bipartite=0)
bnk.add_nodes_from(nutrientList, bipartite=0)
edges = []
weights = []
for crop in countryCrops:
    newEdges = []
    for nutrient in nutrientList:
        weight = np.mean(foodNutrients.loc[foodNutrients['FAO_name']==crop,nutrient])
        weight = weight * 10**6 * 10**(-2) * (1/servingSizes[crop]) / population[population['Country Name']==countryName][year].iloc[0]
        weight = weight * np.mean(production.loc[(production['Area']==countryName)&(production['Item']==crop)]['Y'+year])
        if weight>0.1:
            newEdges.append((crop, nutrient, weight))
            weights.append(weight)
        edges.extend(newEdges)
        bnk.add_weighted_edges_from(newEdges)
    if bnk.degree[crop] == 0:
        bnk.remove_node(crop)
        countryCrops = [x for x in countryCrops if x!=crop]


curve = multicurve_unweighted(bnk,len(countryCrops),1000)

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


















