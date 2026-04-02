import pandas as pd
import networkx as nx
from itertools import combinations


def get_branches(route_name_en):
    """
    Return a list of branches (each branch is a list of station names in order)
    for the given line, based on STATION_ORDER.
    """
    if route_name_en not in STATION_ORDER:
        return []  # no order defined
    return STATION_ORDER[route_name_en]


def build_graph(xlsx_path: str) -> nx.Graph:
    # Load and clean data (same as before)
    df = pd.read_excel(xlsx_path)
    df.columns = [
        "Date",
        "RouteName",
        "RouteNameEn",
        "StationName",
        "StationNameEn",
        "Boarding",
        "Leaving",
        "RecordedDate",
    ]
    df = df.dropna(subset=["RouteNameEn", "StationNameEn"])
    df["StationNameEn"] = df["StationNameEn"].str.replace("\u2019", "'", regex=False)
    df["RouteNameEn"] = df["RouteNameEn"].str.replace("\u2019", "'", regex=False)
    df["StationNameEn"] = df["StationNameEn"].str.strip()
    agg = df.groupby(["RouteName", "RouteNameEn", "StationNameEn"], as_index=False)[
        ["Boarding", "Leaving"]
    ].sum()

    # Build graph
    G = nx.Graph()

    # Add nodes with traffic data
    for _, row in agg.iterrows():
        node_id = f"{row['StationNameEn']} [{row['RouteNameEn']}]"
        G.add_node(
            node_id,
            station=row["StationNameEn"],
            line=row["RouteNameEn"],
            route_korean=row["RouteName"],
            boarding=int(row["Boarding"]),
            leaving=int(row["Leaving"]),
            weight=int(row["Boarding"]) + int(row["Leaving"]),
        )

    # In-line edges: process each branch separately
    for line, group in agg.groupby("RouteNameEn"):
        branches = get_branches(line)
        if not branches:
            # Fallback: use CSV order as a single branch
            stations = group["StationNameEn"].tolist()
            branches = [stations]

        for branch in branches:
            # Filter only stations that exist in the data for this line
            present = [s for s in branch if s in group["StationNameEn"].values]
            if len(present) < 2:
                continue
            # Add edges between consecutive stations
            for i in range(len(present) - 1):
                u = f"{present[i]} [{line}]"
                v = f"{present[i+1]} [{line}]"
                if G.has_node(u) and G.has_node(v):
                    G.add_edge(
                        u,
                        v,
                        type="in_line",
                        line=line,
                        weight=G.nodes[u]["weight"] + G.nodes[v]["weight"],
                    )
            # If it's a circular line (first and last station of the branch are the same),
            # also connect the last to the first.
            if len(present) > 1 and branch[0] == branch[-1]:
                u = f"{present[-1]} [{line}]"
                v = f"{present[0]} [{line}]"
                if G.has_node(u) and G.has_node(v) and not G.has_edge(u, v):
                    G.add_edge(
                        u,
                        v,
                        type="in_line",
                        line=line,
                        weight=G.nodes[u]["weight"] + G.nodes[v]["weight"],
                    )

    # Transfer edges: connect all nodes that share the same station name
    station_nodes = {}
    for node in G.nodes:
        sname = G.nodes[node]["station"]
        station_nodes.setdefault(sname, []).append(node)

    for sname, nodes in station_nodes.items():
        if len(nodes) > 1:
            for u, v in combinations(nodes, 2):
                G.add_edge(
                    u,
                    v,
                    type="transfer",
                    weight=G.nodes[u]["weight"] + G.nodes[v]["weight"],
                )

    return G


def main():
    xlsx_path = "Data_Subway_December.xlsx"
    G = build_graph(xlsx_path)

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Example: top 10 stations by combined traffic
    top = sorted(G.nodes(data=True), key=lambda x: x[1]["weight"], reverse=True)[:10]
    for node, data in top:
        print(f"{node}: {data['weight']:,} passengers")

    nx.write_graphml(G, "seoul_subway_graph.graphml")

    return G


Graph = main()
