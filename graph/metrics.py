# graph/metrics.py
import networkx as nx
import pandas as pd

def basic_properties(G: nx.Graph) -> dict:
    props = {
        "vertices":          G.number_of_nodes(),
        "edges":             G.number_of_edges(),
        "avg_clustering":    nx.average_clustering(G),
        "density":           nx.density(G),
        "is_connected":      nx.is_connected(G) if isinstance(G, nx.Graph) else nx.is_weakly_connected(G),
    }
    return props

def calculate_centralities(G: nx.Graph) -> pd.DataFrame:
    degree  = nx.degree_centrality(G)
    eigen   = nx.eigenvector_centrality(G, max_iter=1000)
    between = nx.betweenness_centrality(G)

    df = pd.DataFrame({
        "user":        list(degree.keys()),
        "degree":      [degree[n]  for n in degree],
        "eigenvector": [eigen[n]   for n in degree],
        "betweenness": [between[n] for n in degree],
    }).sort_values("eigenvector", ascending=False).reset_index(drop=True)

    return df

def node_summary(G: nx.Graph) -> pd.DataFrame:
    centralities = calculate_centralities(G)
    raw_degrees  = dict(G.degree())

    centralities["abs_degree"]  = centralities["user"].map(raw_degrees)
    centralities["clustering"]  = centralities["user"].map(nx.clustering(G))

    return centralities

def print_summary(G: nx.Graph) -> None:
    props = basic_properties(G)

    print("=== Propriedades da rede ===")
    for k, v in props.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Top 10 por eigenvector centrality ===")
    df = node_summary(G)
    print(df.head(10).to_string(index=False))