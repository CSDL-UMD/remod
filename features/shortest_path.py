#!/usr/bin/python3
'''
Calculate the shortest path between subjects and objects in GREC using corpus graph and a trained node2vec model.
Produces a pickled dataframe
'''

import os
import networkx as nx
import pickle
import pandas as pd
from rdflib.extras.external_graph_libs import *
from rdflib import Graph, URIRef, Literal
import rdflib
import re
import gensim
from gensim.models import KeyedVectors
import json
import numpy as np
import logging
import argparse
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from math import acos, pi

def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(description="Calculate shortest paths for GREC subjects/objects")
    now = datetime.datetime.now().strftime("%b-%d-%y")

    parser.add_argument(
        '--weighted',
        dest='dijkstra',
        action='store_true',
        default=False,
        help="Weight the edges, and compute Dijkstra's shortest path"
    )

    parser.add_argument(
        '--directed',
        dest='directed',
        action='store_true',
        default=False,
        help="Use Directed Graph for shortest path calculations"
    )

    parser.add_argument(
        '--n2v-model',
        '-n2v',
        dest="n2v_model",
        type=str,
        default='/data/mjsumpter/nodevectors_ontology_100wl_100nw/models/model-ontology_stitch-100wl-100nw.pkl',
        help="Set filepath to local Node2Vec Model (.pkl), default /data/mjsumpter/nodevectors_ontology_100wl_100nw/models/model-ontology_stitch-100wl-100nw.pkl"
    )

    parser.add_argument(
        '--grec-dir',
        '-grec',
        dest="grec_dir",
        type=str,
        default='/data/mjsumpter/grec/',
        help="Set filepath to GREC jsons, default /data/mjsumpter/grec/"
    )

    parser.add_argument(
        '--rdf-dir',
        '-rdf',
        dest="rdf_dir",
        type=str,
        default='/data/mjsumpter/rdfs/',
        help="Set filepath to GREC jsons, default /data/mjsumpter/rdfs/"
    )

    parser.add_argument(
        '--out-dir',
        '-out',
        dest="out_dir",
        type=str,
        default='/data/mjsumpter/to_be_sorted/',
        help="Set filepath for output dataframe, default /data/mjsumpter/to_be_sorted/"
    )

    parser.add_argument(
        '--experiment-tag',
        '-tag',
        dest="exp_tag",
        type=str,
        default=now,
        help="Set a unique tag for output files from this experiment, default MM-DD-YY"
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()

def generate_out_file(filename, dir, tag):
    """Generates a full output path for a given file

    Arguments:
        filename {str} -- Filename with extension i.e this_graph.txt
        dir {str} -- Directory to save file to
        tag {str} -- Unique Experiment tag to be appended

    Returns:
        str -- Full file path
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

    name = filename.split('.')[0]
    extension = '.' + filename.split('.')[1]

    return dir + name + '-' + tag + extension

def get_rdf_id(rdf_path):
    '''
    Returns rdf id - i.e. /some/path/to/rdf/dob_df8a81jha.rdf -> dob_df8a81jha
    '''
    return re.search('[^\/]*(?=[.][a-zA-Z]+$)', rdf_path).group(0)

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
            for neighbor in nxg.neighbors(node):
                if "dbpedia" in neighbor or "/vn/data/" in neighbor:
                    to_collapse.append((neighbor, node))

    new_G = nxg
    try:
        for node in to_collapse:
            new_G = nx.contracted_nodes(new_G, node[0], node[1])
    except:
        raise
    
    return new_G

def node_to_str(node):
    return node.n3().replace('<', '').replace('>','')

def process_entity(ent_str):
    ent_str = list(map(lambda x:x.lower(), ent_str.split()))

    if len(ent_str) > 1:
        ent_str = [ent_str[0], ent_str[-1]]

    return ent_str

def get_relations(rdf_dir):
    '''
    Returns a list of directory names (should be relations) from master directory of RDFs
    '''
    return [x for x in os.listdir(rdf_dir) if ".csv" not in x]

def get_relation_type(json_filename):
    if "education" in json_filename:
        return "education"
    elif "institution" in json_filename:
        return "institution"
    elif "date_of_birth" in json_filename:
        return "dob"
    elif "place_of_birth" in json_filename:
        return "pob"
    elif "place_of_death" in json_filename:
        return "pod"
    else:
        return "Error"

def get_n2v_embedding(node, n2v):
    """Returns embedding for node using Node2Vec model

    Args:
        node (NetworkX Node): The node to be embedded
        n2v (Node2Vec Model): Model to embed with
    """
    node_str = None
    if type(node) is not rdflib.term.Literal:
        node_str = node_to_str(node)
    else:
        node_str = str(node)
    vector = n2v.wv[node_str]
    return vector

def cosine_distance(x, y):
    """Calculates cosine distance (arccos(cosine_similarity)/pi) for vectors x and y

    Args:
        x (numpy vector)
        y (numpy vector)
    """

    try:
        sim = cosine_similarity(x, y)
        return acos(sim)/pi
    except:
        # sometimes there is a self-loop: This is a bug, and so I weight it significantly
        # so this path won't be travelled in Dijkstra
        return 10000 

def to_weighted_graph(nxg, n2v):
    """Computes weights for each edge in the graph. The weight is the cosine distance between nodes

    Args:
        nxg (NetworkX Graph): Graph to be weighted
        n2v (Node2Vec Model): Model for getting node embeddings
    """

    for node in nxg.nodes():
        node_vec = get_n2v_embedding(node, n2v)
        node_vec = node_vec.reshape(1,-1)
        for neighbor in nxg.neighbors(node):
            neighbor_vec = get_n2v_embedding(neighbor, n2v)
            neighbor_vec = neighbor_vec.reshape(1,-1)
            nxg[node][neighbor]['weight'] = cosine_distance(node_vec, neighbor_vec)

    return nxg

def main():
    ### Initialize Logger ###
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        # datefmt="%H:%M:%S",
        level=logging.INFO
    )

    #########################

    ### Args ###
    args = arg_parse()

    dijkstra = args.dijkstra
    directed = args.directed

    n2v_filepath = args.n2v_model
    grec_dir = args.grec_dir
    rdf_dir = args.rdf_dir
    out_dir = args.out_dir
    tag = args.exp_tag
    ############

    logging.info("Beginning shortest_path.py execution")

    relation_types = get_relations(rdf_dir)
    grec_files = os.listdir(grec_dir)

    n2v_model = KeyedVectors.load(n2v_filepath)
    logging.info("Loaded %s as Node2Vec Model" % n2v_filepath)

    # will hold all final data
    # entry: ['uid', 'subject', 'object', 'relation', 'maj_vote', 'local_short_path', 'struct_short_path']
    data = list()

    for relation_type in relation_types:
        logging.info("Processing %s relation" % relation_type)

        # Get list of .rdf files in directory
        rdfs = os.listdir(rdf_dir + relation_type + '/')
        relations = None

        # find corresponding grec .json, and load into <relations> variable
        grec_filepath = None
        for f in grec_files:
            if get_relation_type(relation_type) in f:
                grec_filepath = grec_dir + f   # append filename to directory path
                with open(grec_filepath, 'r') as f_grec:
                    relations = json.loads(f_grec.read())
                break
        logging.info("Loaded %s" % grec_filepath)

        # for every .rdf in directory
        for rdf in rdfs:
            # generate path
            rdf_path = rdf_dir + relation_type + '/' + rdf

            # set variables to retrieve from grec .json
            rating = None
            subj = None
            obj = None
            db_subj = None
            db_obj = None
            uid = get_rdf_id(rdf_path)

            # get variables from grec .json
            for relation in relations:
                if relation['UID'] == uid:
                    rating = relation['maj_vote']
                    subj = relation['sub']
                    obj = relation['obj']
                    db_subj = relation['dbpedia_sub']
                    db_obj = relation['dbpedia_obj']
                    break

            logging.info(f"Processing {uid}: rating: {rating}, subject: {subj}, object: {obj}")

            # if bad subject/object, skip to next rdf
            if "needs_entry" in subj or "needs_entry" in obj:
                logging.error(f"Bad subject or object, skipping {uid}")

            # prep for string matching
            subj = process_entity(subj)
            obj = process_entity(obj)
            deg_abr = None # this is for trying to capture "Education" degree nodes

            # get just year if object is a date
            if get_relation_type(relation_type) is 'dob':
                obj = obj[0].split('-')[0]
            # Attempt to capture degree abbreviation: ie. Translate Master of Science into m.s
            if get_relation_type(relation_type) is 'education':
                try:
                    deg_abr = f"{obj[0][0]}.{obj[-1][0]}"
                    deg_abr = deg_abr.lower()
                    print(f"Object2: {deg_abr}")
                except:
                    pass

            # Parse graphs, remove VN tags, collapse nodes, and undirect graph
            try:
                graph = Graph()
                graph.parse(rdf_path)
                # graph = remove_vn_tags(graph)
                graph = append_rdf_ids(graph, uid)
                nx_graph = rdflib_to_networkx_multidigraph(graph)
                nx_graph = collapse_fred_nodes(nx_graph)
                # nx_graph = nx_graph.to_undirected() # returns Multigraph object
                if directed:
                    nx_graph = nx.DiGraph(nx_graph)
                else:
                    nx_graph = nx.Graph(nx_graph)
            except Exception as e:
                logging.error(f"Could not generate graphs for {uid}.")
                logging.error(e.__doc__)
                continue

            if dijkstra:
                # Calculate weight for all edges
                try:
                    nx_graph = to_weighted_graph(nx_graph, n2v_model)
                except Exception as e:
                    logging.error(f"Could not weight graph {uid}")
                    logging.error(e.__doc__)
                    continue


            sub_node = None
            obj_node = None

            #### Would like to push these three for loops into one, but priority needs to be maintained
            # First pass through nodes - do any match db_subj or db obj?
            for node in nx_graph.nodes():
                if db_subj in node:
                    sub_node = db_subj
                if db_obj in node:
                    obj_node = db_obj

            # Find nodes that contain object and subject (Second Pass)
            for node in nx_graph.nodes():
                extracted = None   # captures tag of ontology node, capturing its value
                if sub_node is not None and obj_node is not None:
                    # if both found, exit loop
                    break
                if "fred" not in node:
                    extracted = node.split('/')[-1].lower()
                else:
                    extracted = node.split("#")[-1].lower()
                if all(word in extracted for word in subj):
                    logging.info(f"Subject node: {node}")
                    sub_node = node
                if all(word in extracted for word in obj):
                    logging.info(f"Object node: {node}")
                    obj_node = node
                if any(word in extracted for word in subj) and sub_node is None:
                    # more liberal - if no match for full name, try any word in name
                    print(f"Subject node: {node}")
                    sub_node = node
                if any(word in extracted for word in obj) and obj_node is None:
                    # Same as above, but for logic
                    print(f"Object node: {node}")
                    obj_node = node
                if deg_abr is not None:
                    # if "Education", try to match degree abbreviation
                    if deg_abr in extracted and obj_node is None:
                        print(f"Object node: {node}")
                        obj_node = node

            # Final pass, match any words in degree description to an object node - ie 'honorary degree' matches 'degree' or 'honorary doctorate'
            for node in nx_graph.nodes():
                extracted = None
                if "fred" not in node:
                    extracted = node.split('/')[-1].lower()
                else:
                    extracted = node.split("#")[-1].lower()
                if deg_abr is not None:
                    if any(word in extracted for word in obj) and obj_node is None:
                        print(f"Object node: {node}")
                        obj_node = node


            # if subject or object could not be found, skip
            if sub_node is None:
                logging.error(f"Couldn't find subject in graph: {subj}")
                continue
            if obj_node is None:
                logging.error(f"Couldn't find object in graph: {obj}")
                continue

            # shortest path between subject and object (as a list)
            try:
                if dijkstra:
                    shortest_path = nx.dijkstra_path(nx_graph, obj_node, sub_node)
                else:
                    shortest_path = nx.shortest_path(nx_graph, obj_node, sub_node)
            except Exception as e:
                logging.error(f"There is no path found between {obj_node} and {sub_node}. Relation: {uid}")
                continue

            # Calculate normalized vectors for path
            ## vector_final holds sum of all vectors in path
            vector_final = None

            ## get vector for every node and add them
            for node in shortest_path:
                vector = get_n2v_embedding(node, n2v_model)
                if vector_final is None:
                    # for first vector
                    vector_final = vector
                else:
                    vector_final = vector_final + vector

            # if these are none, there was an error. Skip
            if vector_final is None:
                logging.error("Issue with producing embeddings...")
                continue

            # Normalize vector
            n2v_norm = np.linalg.norm(vector_final)
            vector_final = vector_final/n2v_norm

            # append new entry to list
            new_entry = [uid, subj, obj, relation_type, rating, vector_final]
            data.append(new_entry)


            logging.info(f"Finished processing {uid}")

        logging.info("Finished processing %s relation" % relation_type)

    df = pd.DataFrame(data, columns = ['UID', 'Subject', 'Object', 'Relation', 'Maj_Vote', 'Short_Path'])
    out_file = generate_out_file("shortest_path_df.pkl", out_dir, tag)
    df.to_pickle(out_file)
    logging.info(f"Shortest paths written to {out_file}")

    logging.info("Completed shortest_path.py execution")

if __name__ == "__main__":
    main()