import networkx as nx
from typing import Type

def collapse_fred_nodes(nxg: Type[nx.Graph]) -> Type[nx.Graph]:
    """Collapses FRED nodes that point to DBPedia or VerbNet Nodes

    Arguments:
        nxg {NetworkX Graph Object} -- Graph to be modified

    Returns:
        NetworkX Graph Object -- Modified NetworkX Graph
    """
    to_collapse = [] # nodes that need collapsing
    for node in nxg.nodes():
        if "fred" in node:
            ontology_neighbors = []
            for neighbor in nxg.neighbors(node):
                if "dbpedia" in neighbor or "/vn/data/" in neighbor:
                    ontology_neighbors.append((neighbor, node))
            if len(ontology_neighbors) == 1:
                # only collapse if there in one ontology neighbor, otherwise leave as is
                to_collapse.append(ontology_neighbors[0])

    new_G = nxg
    try:
        for node in to_collapse:
            new_G = nx.contracted_nodes(new_G, node[0], node[1])
    except:
        raise
    
    return new_G

def node_to_str(node: Type[nx.Graph.node]) -> str:
    return node.n3().replace('<', '').replace('>','')

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