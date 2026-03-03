import pandas as pd
import networkx as nx

df = pd.read_csv('CARD_SUBWAY_MONTH_202512.csv', encoding='utf-8-sig', usecols=[0,1,2,3,4,5,6])
df.columns = ['Date','RouteName','RouteNameEn','StationName','StationNameEn','Boarding','Leaving']

# Aggregate across dates (sum boardings/leavings per station+line)
agg = df.groupby(['RouteNameEn','StationNameEn'])[['Boarding','Leaving']].sum().reset_index()

G = nx.Graph()

# Add nodes: one per (station, line) pair
for _, row in agg.iterrows():
    node_id = f"{row['StationNameEn']} [{row['RouteNameEn']}]"
    G.add_node(node_id, station=row['StationNameEn'], line=row['RouteNameEn'],
               boarding=row['Boarding'], leaving=row['Leaving'])

# Add in-line edges (consecutive stations per line)
for line, group in agg.groupby('RouteNameEn'):
    stations = group['StationNameEn'].tolist()  # order from file = line order
    for i in range(len(stations) - 1):
        u = f"{stations[i]} [{line}]"
        v = f"{stations[i+1]} [{line}]"
        G.add_edge(u, v, type='in_line', line=line)

# Add transfer edges between nodes that share a station name
from itertools import combinations
station_nodes = {}
for node in G.nodes:
    sname = G.nodes[node]['station']
    station_nodes.setdefault(sname, []).append(node)

for sname, nodes in station_nodes.items():
    if len(nodes) > 1:
        for u, v in combinations(nodes, 2):
            G.add_edge(u, v, type='transfer')