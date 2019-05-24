# Makes bipartite crop-nutrient graph, for country specified by argument.
# Crop data from 2017 is used

import pandas as pd
import glob
import csv
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import sys
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


countryName = sys.argv[1]
production = pd.read_csv('data/Production_Crops_E_All_Data.csv',encoding = "ISO-8859-1")
names_table_info = pd.read_csv('data/food_names_nutrients_table.csv')


countryCrops = [x for x in list(set(production.loc[(production['Area']==countryName)&(production['Y2017']>0),'Item'].to_list())) if x in names_table_info['FAO_name'].tolist()]
nutrientList = list(names_table_info)[3:]
bnk = nx.OrderedGraph()
bnk.add_nodes_from(countryCrops, bipartite=0)
bnk.add_nodes_from(nutrientList, bipartite=0)

edges = []
for crop in countryCrops:
    for nutrient in nutrientList:
        if np.mean(names_table_info.loc[names_table_info['FAO_name']==crop,nutrient])>0:
            edges.append((crop, nutrient))

bnk.add_edges_from(edges)

fig, ax = plt.subplots(figsize=(10,.4*len(countryCrops)))

pos=nx.drawing.layout.bipartite_layout(bnk,countryCrops)
nx.draw_networkx(bnk,pos,with_labels=False,edge_color='green')
for crop in countryCrops:
    plt.text(pos[crop][0]-.05,pos[crop][1],s=crop,color='k',fontsize=20,ha='right')

for nutrient in nutrientList:
    plt.text(pos[nutrient][0]+.05,pos[nutrient][1],s=nutrient,color='k',fontsize=20,ha='left')



plt.axis('off')

plt.title('Crop-Nutrient Network: '+countryName,fontsize=30,y=.96)
plt.savefig(countryName.lower()+'_bipartite-narrow.pdf',transparent=True,bbox_inches='tight')

