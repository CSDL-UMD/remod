from node2vec import Node2Vec
from rdflib.extras.external_graph_libs import *
from rdflib import Graph, URIRef, Literal
import networkx as nx
import numpy as np
import os
import pickle

# Node2Vec Constants
dimensions = 128
walk_length = 39
num_walks = 200
workers = 4
window = 10
min_count = 1
batch_words = 4

# join on dbpedia entities
# do nodes get converted to integers on the backend?
# networkx (union/disjoin union)
# size of final network
# get recommendation from node2vec paper for walk length/number/etc

rdf_dir = '/data/mjsumpter/google_relation_extraction_corpus/rdf/'
relations = [x for x in os.listdir(rdf_dir) if '.csv' not in x]

relation_vector_dict = dict()

for relation in relations:
    
    relation_dir = rdf_dir + relation + '/'
    rdf_files = [x for x in os.listdir(relation_dir) if '.rdf' in x]
    
    relation_vector_dict[relation] = list()
    
    for rdf_file in rdf_files:
        rdf_path = relation_dir + rdf_file
        # Extract RDF Graph
        graph = Graph()
        try:
            graph.parse(rdf_path)
        except Exception as e:
            print("graph.parse(): Line 35")
            print("%s of relation type %s failed" % (rdf_file, relation))
            print(e.__doc__)           
            continue
        try:
            # Convert to networkx graph
            nx_graph = rdflib_to_networkx_multidigraph(graph)
        except Exception as e:
            print("rdflib_to_networkx_multidigraph(graph): Line 43")
            print("%s of relation type %s failed" % (rdf_file, relation))
            print(e.__doc__)           
            continue
        try:
            # Train and Fit Node2Vec
            node2vec = Node2Vec(nx_graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        except Exception as e:
            print("Node2Vec(): Line 50")
            print("%s of relation type %s failed" % (rdf_file, relation))
            print(e.__doc__)           
            continue
        try:    
            model = node2vec.fit(window=window, min_count=1, batch_words=4)
        except Exception as e:
            print("node2vec.fit(): Line 58")
            print("%s of relation type %s failed" % (rdf_file, relation))
            print(e.__doc__)           
            continue
        # Get number of nodes
        num_nodes = len(model.wv.vocab)
        # Get a list of all Vectors
        vec_list = np.empty([num_nodes, dimensions])
        for idx, key in enumerate(model.wv.vocab):
            vec_list[idx] = model.wv[key]
        # Average all nodes, and append to the list of graph vectors for the relation    
        relation_vector_dict[relation].append(np.mean(vec_list, axis=0))

# dump dictionary for analysis
output_file = '/data/mjsumpter/google_relation_extraction_corpus/relation_vec_avg.pkl'

with open(output_file, 'wb') as relation_vector_file:
    pickle.dump(relation_vector_dict, relation_vector_file)