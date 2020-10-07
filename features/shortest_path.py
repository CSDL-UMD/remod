"""
Calculate the shortest path between subjects and objects in GREC using corpus graph and a trained node2vec model.
Produces a pickled dataframe.

Currently only works for GREC and GREC Relations
"""

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
import argparse
import datetime
from typing import Type
from sklearn.metrics.pairwise import cosine_similarity
from math import acos, pi
from utils.rdf import append_rdf_ids, remove_vn_tags, get_rdfGraph
from utils.nx_helpers import collapse_fred_nodes, node_to_str
from utils.file import generate_out_file
from utils.n2v_helpers import (
    get_n2v_vector,
    get_nodevectors_vector,
    load_n2v_model,
    load_nodevectors_model,
)


def process_entity(ent_str):
    ent_str = list(map(lambda x: x.lower(), ent_str.split()))

    if len(ent_str) > 1:
        ent_str = [ent_str[0], ent_str[-1]]

    return ent_str


def cosine_distance(x: Type[np.ndarray], y: Type[np.ndarray]) -> float:
    """Calculates cosine distance (arccos(cosine_similarity)/pi) for vectors x and y

    Args:
        x (numpy vector)
        y (numpy vector)
    """

    try:
        sim = cosine_similarity(x, y)
        return acos(sim) / pi
    except:
        # sometimes there is a self-loop: This is a bug, and so I weight it significantly
        # so this path won't be travelled in Dijkstra
        return 10000


def to_weighted_graph(nxg, n2v, nodevectors: bool = False):
    """Computes weights for each edge in the graph. The weight is the cosine distance between nodes

    Args:
        nxg (NetworkX Graph): Unweighted NetworkX Graph
        n2v (Node2Vec model): Model for generating node embeddings
        nodevectors (bool, optional): If true, use nodevectors rather than Node2Vec. Defaults to False.

    Returns:
        NetworkX Graph: Returns nxg, but weighted
    """

    for node in nxg.nodes():
        node_vec = (
            get_nodevectors_vector(n2v, node)
            if nodevectors
            else get_n2v_vector(n2v, node)
        )
        node_vec = node_vec.reshape(1, -1)
        for neighbor in nxg.neighbors(node):
            neighbor_vec = (
                get_nodevectors_vector(n2v, neighbor)
                if nodevectors
                else get_n2v_vector(n2v, neighbor)
            )
            neighbor_vec = neighbor_vec.reshape(1, -1)
            nxg[node][neighbor]["weight"] = cosine_distance(node_vec, neighbor_vec)

    return nxg


def generate_sp_df(
    n2v_model_file: str,
    snippets: str,
    rdf_dir: str,
    out_dir: str,
    tag: str,
    weighted: bool = False,
    directed: bool = False,
    existing: str = None,
) -> None:
    """Generates a dataframe of shortest path vectors between two nodes.

    Args:
        n2v_model_file (str): Path to Node2Vec model
        snippets (str): Path to a .json containing snippets containing relations
        rdf_dir (str): Path to the directory of RDFs corresponding to the snippets
        out_dir (str): The directory that the dataframe will be written to
        tag (str): The experimental tag, to be appended to the output file name
        weighted (bool, optional): Process as weighted graph. Defaults to False.
        directed (bool, optional): Process as directed graph. Defaults to False. 
        existing (str, optional): Filepath to an existing dataframe to append to. Defaults to None.
    """

    now = datetime.datetime.now()
    print("-" * 30)
    print("Beginning shortest_path.generate_sp_df()")
    print("-" * 30)
    print(f"N2V Model: {n2v_model_file}")
    print(f"Snippet File: {snippets}")
    print(f"RDF Dir: {rdf_dir}")

    n2v_model = None
    nv = False
    if "nv" in n2v_model_file:
        n2v_model = load_nodevectors_model(n2v_model_file)
        nv = True
    else:
        n2v_model = load_n2v_model(n2v_model_file)

    data = list()

    # Get list of .rdf files in directory
    rdfs = os.listdir(rdf_dir)
    relations = None
    relation_type = snippets.split("/")[-1].split("_")[0]  # very GREC specific

    # load snippets into <relations> variable
    with open(snippets, "r") as f_grec:
        relations = json.loads(f_grec.read())

    # for every .rdf in directory
    for rdf in rdfs:
        # generate path
        rdf_path = rdf_dir + "/" + rdf

        # set variables to retrieve from grec .json
        rating = None
        subj = None
        obj = None
        db_subj = None
        db_obj = None
        uid = rdf.split(".")[0]

        # get variables from grec .json
        for relation in relations:
            if relation["UID"] == uid:
                rating = relation["maj_vote"]
                subj = relation["sub"]
                obj = relation["obj"]
                db_subj = relation["dbpedia_sub"]
                db_obj = relation["dbpedia_obj"]
                break

        print(f"Processing {uid}: rating: {rating}, subject: {subj}, object: {obj}")

        # if bad subject/object, skip to next rdf
        if "needs_entry" in subj or "needs_entry" in obj:
            print(f"ERROR: Bad subject or object, skipping {uid}")
            continue

        # prep for string matching
        subj = process_entity(subj)
        obj = process_entity(obj)
        deg_abr = None  # this is for trying to capture "Education" degree nodes

        # get just year if object is a date
        if relation_type == "dob":
            obj = obj[0].split("-")[0]
        # Attempt to capture degree abbreviation: ie. Translate Master of Science into m.s
        if relation_type == "education":
            try:
                deg_abr = f"{obj[0][0]}.{obj[-1][0]}"
                deg_abr = deg_abr.lower()
                print(f"Object2: {deg_abr}")
            except:
                pass

        # Parse graphs, remove VN tags, collapse nodes, and undirect graph
        try:
            graph = get_rdfGraph(rdf_path)
            # graph = remove_vn_tags(graph)
            # graph = append_rdf_ids(graph, uid)
            nx_graph = rdflib_to_networkx_multidigraph(graph)
            nx_graph = collapse_fred_nodes(nx_graph)
            nx_graph = nx_graph.to_undirected() # returns Multigraph object
            if directed:
                nx_graph = nx.DiGraph(nx_graph)
            else:
                nx_graph = nx.Graph(nx_graph)
        except Exception as e:
            logging.error(f"Could not generate graphs for {uid}.")
            logging.error(e.__doc__)
            continue

        if weighted:
            # Calculate weight for all edges
            try:
                nx_graph = to_weighted_graph(nx_graph, n2v_model, nv)
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
            extracted = None  # captures tag of ontology node, capturing its value
            if sub_node is not None and obj_node is not None:
                # if both found, exit loop
                break
            if "fred" not in node:
                extracted = node.split("/")[-1].lower()
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
                extracted = node.split("/")[-1].lower()
            else:
                extracted = node.split("#")[-1].lower()
            if deg_abr is not None:
                if any(word in extracted for word in obj) and obj_node is None:
                    print(f"Object node: {node}")
                    obj_node = node

        # if subject or object could not be found, skip
        if sub_node is None:
            print(f"ERROR: Couldn't find subject in graph: {subj}")
            continue
        if obj_node is None:
            print(f"ERROR: Couldn't find object in graph: {obj}")
            continue

        # shortest path between subject and object (as a list)
        try:
            if weighted:
                shortest_path = nx.dijkstra_path(nx_graph, obj_node, sub_node)
            else:
                shortest_path = nx.shortest_path(nx_graph, obj_node, sub_node)
        except Exception as e:
            print(
                f"ERROR: There is no path found between {obj_node} and {sub_node}. Relation: {uid}"
            )
            continue

        # Calculate normalized vectors for path
        ## vector_final holds sum of all vectors in path
        vector_final = None

        ## get vector for every node and add them
        for node in shortest_path:
            vector = (
                get_nodevectors_vector(n2v_model, node)
                if nv
                else get_n2v_vector(n2v_model, node)
            )
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
        vector_final = vector_final / n2v_norm

        # append new entry to list
        new_entry = [uid, subj, obj, relation_type, rating, vector_final]
        data.append(new_entry)

        print(f"Finished processing {uid}")

    df = pd.DataFrame(
        data, columns=["UID", "Subject", "Object", "Relation", "Maj_Vote", "Short_Path"]
    )
    out_file = generate_out_file("sp_df.pkl", out_dir, tag)

    if existing:
        df_existing = pd.read_pickle(existing)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_pickle(out_file)
    print(f"Shortest paths written to {out_file}")
    print("Completed shortest_path.py execution")
    print("-" * 30)


if __name__ == "__main__":
    main()
