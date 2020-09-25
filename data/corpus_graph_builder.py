import os
import networkx as nx
from rdflib.extras.external_graph_libs import *
from rdflib import Graph, URIRef, Literal
import pickle
import datetime
import logging
import re
import argparse
from utils.rdf import get_rdfGraph
from utils.file import get_filename, generate_out_file
from tqdm import tqdm

NOW = datetime.datetime.now().strftime("%y%b%d")

def log_graph_info(id, nxg, log, state):
    """Logs number of nodes to output file

    Arguments:
        id {str} -- File ID/UID
        nxg {NetworkX Graph} -- NetworkX graph to be logged
        log {str} -- Path to .log file
        state {str} -- Before collapse (b), after collapse (a)
    """
    num_nodes = len(nxg.nodes)
    with open(log, 'a+') as f:
        if state == "b":
            f.write("%s nodes before: %d\n" %(id, num_nodes))
        else:
            f.write("%s nodes after: %d\n" %(id, num_nodes))

def collapse_fred_nodes(nxg):
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

def append_rdf_ids(rdflib_g, uid):
    """Appends an RDF UID to all FRED nodes in RDF

    Arguments:
        rdflib_g {RdfLib Graph Object} -- Rdflib graph object to be altered
        uid {str} -- Unique ID to be appended

    Returns:
        RdfLib Graph Object -- Altered Graph
    """
    for s,p,o in rdflib_g:
        new_o = o
        new_s = s
        if "fred" in o:
            new_o = o + '-' + uid
        if "fred" in s:
            new_s = s + '-' + uid
    
        rdflib_g.remove((s,p,o))
        rdflib_g.add((new_s, p, new_o))
    return rdflib_g

def remove_vn_tags(rdflib_g):
    """Removes VN Tags i.e http://www.ontologydesignpatterns.org/ont/vn/data/Direct <_392939>

    Arguments:
        rdflib_g {RdfLib Graph Object} -- Rdflib graph object to be altered

    Returns:
        RdfLib Graph Object -- Altered Graph
    """
    for s,p,o in rdflib_g:
        new_o = o
        new_s = s
        if "/vn/data/" in s:
            to_remove = re.search('_\d+', s).group(0)
            new_s = s.replace(to_remove, "")
            new_s = URIRef(new_s)
        if "/vn/data/" in o:
            to_remove = re.search('_\d+', o).group(0)
            new_o = o.replace(to_remove, "")
            new_o = URIRef(new_o)
        rdflib_g.remove((s,p,o))
        rdflib_g.add((new_s, p, new_o))
    return rdflib_g

def create_graph(rdf_dir: str, out_dir: str, fred: bool, append: bool, tag: str = NOW, existing: str = None) -> None:
    """
    rdf_dir: Directory where RDFs are located
    out_dir: Directory to output graph and log
    fred: If True, leave FRED nodes intact
    append: If True, append file ids to FRED nodes
    tag (optional): a unique tag for the output files. Defaults to current time
    existing (option): path to an existing networkx graph to append to. Should be pickled format. Default None
    """

    full_graph = None

    # Initialize structure that will become final output graph
    if existing:
        full_graph = nx.read_gpickle(existing)
    else:
        full_graph = nx.MultiGraph()

    rdf_files = [x for x in os.listdir(rdf_dir) if '.rdf' in x]

    # for every rdf file
    for rdf_file in tqdm(rdf_files):

        print(f"\nParsing {rdf_file}")

        rdf_path = rdf_dir + rdf_file
        
        graph = None
        ### Parse RDF Graph
        try:
            graph = get_rdfGraph(rdf_path)
        except Exception as e:
            logging.error("Failed to parse: %s" % rdf_file)
            logging.error(e.__doc__)
            continue
        
        # Append unique RDF ids to FRED nodes (limits how much graph is combined)
        if append:
            uid = get_filename(rdf_path)
            graph = append_rdf_ids(graph, uid)

        # Make NetworkX Graph
        try:
            nx_graph = rdflib_to_networkx_multidigraph(graph) # rdf ->networkx
        except Exception as e:
            logging.error("Failed to parse RDF to NetworkX: %s" % rdf_file)
            logging.error(e.__doc__)
            continue

        # Collapse out FRED nodes
        if not fred:
            try:
                nx_graph = collapse_fred_nodes(nx_graph)
            except Exception as e:
                logging.error("Failed to collapse FRED Nodes: %s" % rdf_file)
                logging.error(e.__doc__)
                continue

        # Add new graph to corpus graph
        try:
            full_graph = nx.compose(full_graph, nx_graph)
        except Exception as e:
            logging.error("Failed to append %s to corpus graph" % rdf_file)
            logging.error(e.__doc__)
            continue

    out_graph = generate_out_file('corpus_graph.pkl', out_dir, tag)
    nx.write_gpickle(full_graph, out_graph)

    print(f"Completed appending {rdf_dir}")
