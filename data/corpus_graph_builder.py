import os
import networkx as nx
from rdflib.extras.external_graph_libs import *
from rdflib import Graph, URIRef, Literal
import pickle
import datetime
import logging
import re
import argparse
from utils.rdf import get_rdfGraph, append_rdf_ids, remove_vn_tags
from utils.file import get_filename, generate_out_file
from utils.nx_helpers import collapse_fred_nodes
from tqdm import tqdm

NOW = datetime.datetime.now().strftime("%y%m%d")

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
