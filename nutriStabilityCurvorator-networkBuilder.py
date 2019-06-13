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


countryCrops = [x for x in list(set(production.loc[(production['Area']==countryName)&(production['Y'+str(year)]>0),'Item'].to_list())) if x in list(servingSizes.keys())]
nutrientList = list(foodNutrients)[3:-1]
bnk = nx.OrderedGraph()
bnk.add_nodes_from(countryCrops, bipartite=0)
bnk.add_nodes_from(nutrientList, bipartite=0)

edges = []
weights = []
for crop in countryCrops:
    for nutrient in nutrientList:
        weight = np.mean(foodNutrients.loc[names_table_info['FAO_name']==crop,nutrient])
        if weight>thresh:
            edges.append((crop, nutrient))
            weights.append(weight)

bnk.add_edges_from(edges)

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

plt.close()


fig, ax = plt.subplots(figsize=(10,.4*len(countryCrops)))

pos=nx.drawing.layout.bipartite_layout(bnk,countryCrops)
# nx.draw_networkx(bnk,pos,with_labels=False,edge_color='green',width=[100*np.log10(w+1) for w in weights])
nx.draw_networkx(bnk,pos,with_labels=False,edge_color='green',width=[20*w for w in weights])
for crop in countryCrops:
    plt.text(pos[crop][0]-.05,pos[crop][1],s=crop,color='k',fontsize=20,ha='right')

for nutrient in nutrientList:
    plt.text(pos[nutrient][0]+.05,pos[nutrient][1],s=nutrient,color='k',fontsize=20,ha='left')



plt.axis('off')

plt.title('Crop-Nutrient Network: '+countryName+', '+year,fontsize=25,y=.96)
plt.savefig(countryName.lower()+'-'+year+'_bipartite-weighted.pdf',transparent=True,bbox_inches='tight')

plt.close()
















