# graph/viz.py
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt

def _node_size(centrality: dict, node: str, scale: float = 40.0) -> float:
    return centrality.get(node, 0) * scale + 8

def _node_color(centrality: dict, node: str) -> str:
    v = centrality.get(node, 0)
    if v > 0.6:
        return "#534AB7"   # purple — highly central
    elif v > 0.3:
        return "#1D9E75"   # green  — mid centrality
    else:
        return "#888780"   # gray   — peripheral

def plot_network(
    G: nx.Graph,
    centrality: dict,
    title: str = "Discord Network",
    output: str = "network.html",
) -> None:
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=isinstance(G, nx.DiGraph),
    )

    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=120,
        spring_strength=0.08,
        damping=0.4,
    )

    for node in G.nodes():
        size       = _node_size(centrality, node)
        color      = _node_color(centrality, node)
        eigen_val  = centrality.get(node, 0)
        degree     = G.degree(node)

        net.add_node(
            node,
            label=node if size > 20 else "",   # only show label on larger nodes
            title=f"({node}) - (grau: {degree}) - (eigenvector: {eigen_val:.3f})",
            size=size,
            color=color,
        )

    for u, v in G.edges():
        net.add_edge(u, v, color="rgba(255,255,255,0.15)", width=1)

    net.show_buttons(filter_=["physics"])
    net.save_graph(output)
    print(f"Rede salva em: {output}")

def plot_degree_distribution(G: nx.Graph, output: str = "degrees.png") -> None:
    degrees = [d for _, d in G.degree()]

    plt.figure(figsize=(8, 4))
    plt.hist(degrees, bins=20, color="#534AB7", edgecolor="white", linewidth=0.5)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree distribution")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Distribuição salva em: {output}")