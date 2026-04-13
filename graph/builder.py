# graph/builder.py
import pandas as pd
import networkx as nx
from config import root

def load_messages(server: str, channel: str) -> pd.DataFrame:
    path = f"{root}/channel_history/{server}/{channel}.csv"
    return pd.read_csv(path)

def extract_edges(df: pd.DataFrame) -> list[tuple]:
    edges = []
    id_to_author = dict(zip(df['id'], df['autor']))

    for _, row in df.iterrows():
        # Source 1: mentions
        if pd.notna(row['mentioned_users']) and row['mentioned_users'] != '':
            for mentioned in row['mentioned_users'].split(','):
                mentioned = mentioned.strip()
                if mentioned and mentioned != row['autor']:
                    edges.append((row['autor'], mentioned))

        # Source 2: replies
        if pd.notna(row['reply_to_id']) and row['reply_to_id'] != '':
            original_author = id_to_author.get(int(row['reply_to_id']))
            if original_author and original_author != row['autor']:
                edges.append((row['autor'], original_author))

    return edges

def build_graph_from_edges(edges: list[tuple]) -> tuple[nx.DiGraph, nx.Graph]:
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G_undir = G.to_undirected()
    return G, G_undir

def build_graph(server: str, channel: str) -> tuple[nx.DiGraph, nx.Graph]:
    df = load_messages(server, channel)
    edges = extract_edges(df)

    print(f"Total de arestas brutas: {len(edges)}")

    G, G_undir = build_graph_from_edges(edges)

    print(f"Vértices: {G.number_of_nodes()}")
    print(f"Arestas: {G.number_of_edges()}")

    return G, G_undir