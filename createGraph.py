import pandas as pd
import networkx as nx
from itertools import combinations
from seoul_station_orders import STATION_ORDER


def get_sorted_stations(route_name_en, df_agg):
    """
    Return stations for a given RouteNameEn in correct line order.
    Falls back to original CSV order if line not in STATION_ORDER.
    """
    group = df_agg[df_agg["RouteNameEn"] == route_name_en]["StationNameEn"].tolist()

    if route_name_en not in STATION_ORDER:
        print(f"[WARN] No order defined for '{route_name_en}', using CSV order.")
        return group

    reference = STATION_ORDER[route_name_en]
    ordered = [s for s in reference if s in group]
    missing = [s for s in group if s not in reference]
    if missing:
        print(f"[WARN] {route_name_en}: stations not in reference order: {missing}")
        ordered += missing

    return ordered


def build_graph(xlsx_path: str) -> nx.Graph:
    # ── Load & aggregate ────────────────────────────────────────────────────
    df = pd.read_excel(xlsx_path)
    # Expected columns: Date, RouteName, RouteNameEn, StationName, StationNameEn,
    #                   Boarding, Leaving, RecordedDate
    df.columns = ["Date", "RouteName", "RouteNameEn",
                  "StationName", "StationNameEn",
                  "Boarding", "Leaving", "RecordedDate"]
    df = df.dropna(subset=["RouteNameEn", "StationNameEn"])

    df["StationNameEn"] = df["StationNameEn"].str.replace("\u2019", "'", regex=False)
    df["RouteNameEn"] = df["RouteNameEn"].str.replace("\u2019", "'", regex=False)

    # Sum boarding/leaving across all days in the file
    agg = (df.groupby(["RouteName", "RouteNameEn", "StationNameEn"],
                      as_index=False)[["Boarding", "Leaving"]].sum())

    # ── Build graph ─────────────────────────────────────────────────────────
    G = nx.Graph()

    # Nodes: one per (station, line) pair; boarding+leaving = node weight
    for _, row in agg.iterrows():
        node_id = f"{row['StationNameEn']} [{row['RouteNameEn']}]"
        G.add_node(
            node_id,
            station=row["StationNameEn"],
            line=row["RouteNameEn"],
            route_korean=row["RouteName"],
            boarding=int(row["Boarding"]),
            leaving=int(row["Leaving"]),
            weight=int(row["Boarding"]) + int(row["Leaving"]),  # combined traffic
        )

    # In-line edges: consecutive stations along each line
    for line, group in agg.groupby("RouteNameEn"):
        stations = get_sorted_stations(line, agg)
        for i in range(len(stations) - 1):
            u = f"{stations[i]} [{line}]"
            v = f"{stations[i+1]} [{line}]"
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v, type="in_line", line=line)

    # Transfer edges: connect all nodes that share the same station name
    station_nodes = {}
    for node in G.nodes:
        sname = G.nodes[node]["station"]
        station_nodes.setdefault(sname, []).append(node)

    for sname, nodes in station_nodes.items():
        if len(nodes) > 1:
            for u, v in combinations(nodes, 2):
                G.add_edge(u, v, type="transfer")

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


if __name__ == "__main__":
    main()