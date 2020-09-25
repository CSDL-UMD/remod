import networkx as nx
from typing import Type

def uri_to_str(G: Type[nx.Graph]) -> Type[nx.Graph]:
    """When given a graph of RDF URI nodes, converts nodes from URI to str

    Args:
        G (Type[nx.Graph]): NetworkX Graph of URI Nodes

    Returns:
        Type[nx.Graph]: Equivalent of G, but with string nodes
    """

    mapping = dict()
    for node in G.nodes:
        mapping[node] = str(node)

    return nx.relabel_nodes(G, mapping)