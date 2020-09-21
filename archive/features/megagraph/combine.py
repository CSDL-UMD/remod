# combines all rdf relation graphs into one single, ultra graph

import os
import networkx as nx
from rdflib.extras.external_graph_libs import *
from rdflib import Graph, URIRef, Literal
import pickle
from datetime import datetime

print("Started at %s" % datetime.now())

rdf_dir = '/data/mjsumpter/google_relation_extraction_corpus/rdf/'

relations = [x for x in os.listdir(rdf_dir) if ".csv" not in x]



for relation in relations:

    full_graph = nx.MultiGraph()

    relation_dir = rdf_dir + relation + '/'
    rdf_files = [x for x in os.listdir(relation_dir) if '.rdf' in x]

    for rdf_file in rdf_files:
        rdf_path = relation_dir + rdf_file

        graph = Graph()
        try:
            graph.parse(rdf_path)
        except Exception as e:
            print(e.__doc__)
            continue
        try:
            nx_graph = rdflib_to_networkx_multidigraph(graph)
            full_graph = nx.compose(full_graph, nx_graph)
        except Exception as e:
            print(e.__doc__)
            continue

    out_graph = rdf_dir + relation + '-ultragraph.pkl'
    nx.write_gpickle(full_graph, out_graph)

    print("Finished %s at %s" % (relation, datetime.now()))

print("Finished at %s" % datetime.now())