from rdflib.extras.external_graph_libs import *
from rdflib import Graph, URIRef, Literal
import rdflib
import pandas as pd
import json
from utils.rdf import append_rdf_ids, remove_vn_tags, get_rdfGraph
from utils.file import generate_out_file
import datetime
import os
import networkx as nx
from utils.nx_helpers import collapse_fred_nodes
import re


DEGREE_ABRS = {
    "Bachelor of Science": ["b.s", "b.sc", "s.b"],
    "Doctor of Philosophy": ["d.phil", "ph.d"],
    "Bachelor of Arts": ["a.b", "b.a", "b.s"],
    "Bachelor of Engineering": ["b.e", "b.eng", "b.s.e", "b.s.e.e"],
    "Master of Science": ["m.s", "m.sc", "s.m"],
    "Bachelor of Technology": ["b.tech"],
    "Master of Arts": ["m.a"],
    "Master's Degree": ["m.s"],
    "Master of Philosophy": ["m.phil"],
    "Legum Doctor": ["ll.d"],
    "Bachelor's degree": ["a.b"],
    "Master of Business Administration": ["m.b.a"],
    "Bachelor of Laws": ["b.l", "l.l.b", "ll.b"],
    "Dental degree": ["d.d.s"],
    "Doctor of Law": ["ll.d"],
    "Juris Doctor": ["j.d"],
    "Doctor of Medicine": ["m.d"],
    "Bachelor of Music": ["b.m", "b.mus", "mus.b"],
    "Bachelor of Education": ["b.ed"],
    "Doctor of Letters": ["d.litt", "litt.d"],
    "Doctor of Education": ["ed.d"],
    "Master of Social Work": ["m.s.w"],
    "Bachelor of Philosophy": ["ph.b"],
    "Bachelor of Fine Arts": ["b.f.a"],
    "Business administration": ["b.b.a"],
    "Doctor of Science": ["d.sc"],
    "Doctor of Juridical Science": ["s.j.d"],
    "Bachelor of Electrical Engineering": ["b.e.e"],
    "Bachelor of Theology": ["th.b"],
    "Doctor of Humane Letters": ["l.h.d"],
    "Master of Laws": ["ll.m"],
    "Bachelor of Commerce": ["b.comm"],
    "Doctor of Divinity": ["d.d"],
}


def process_entity(ent_str):
    ent_str = list(map(lambda x: x.lower(), ent_str.split()))

    if len(ent_str) > 1:
        ent_str = [ent_str[0], ent_str[-1]]

    return ent_str


def find_terminal_nodes(
    rdf_path: str, subj: str, obj: str, db_subj: str, db_obj: str
) -> tuple:
    """Find the selected terminal nodes in a provided rdf graph

    Args:
        rdf_path (str): Path to the rdf graph
        subj (str): string representing subject
        obj (str): string representing object
        db_subj (str): the dbpedia node for the subject
        db_obj (str): the dbpedia node for the object

    Returns:
        tuple: Returns (subject, object) nodes
    """

    relation = rdf_path.split("/")[-1].split("_")[0]

    deg_abrs = None  # this is for trying to capture "Education" degree nodes
    # Attempt to capture degree abbreviation: ie. Translate Master of Science into m.s
    if relation == "e":
        try:
            deg_abrs = DEGREE_ABRS[obj]
        except:
            pass

    # prep for string matching
    subj = process_entity(subj)
    obj = process_entity(obj)
    
    # get just year if object is a date
    if relation == "dob":
        obj = [obj[0].split("-")[0]]

    sub_node = None
    obj_node = None

    try:
        graph = get_rdfGraph(rdf_path)
        graph = rdflib_to_networkx_multidigraph(graph)
        graph = collapse_fred_nodes(graph)
        graph = graph.to_undirected()
    except:
        return ("Not Found", "Not Found")

    # Build list of nodes
    nodes = list(set(graph.nodes()))

    # First pass through nodes - do any match db_subj or db obj?
    for node in nodes:
        if db_subj in node:
            sub_node = node
        if db_obj in node:
            obj_node = node

    # Find nodes that contain object and subject (Second Pass)
    for node in nodes:
        extracted = None  # captures tag of ontology node, capturing its value
        if sub_node is not None and obj_node is not None:
            # if both found, exit loop
            break
        if "fred" not in node:
            extracted = node.split("/")[-1].lower()
        else:
            extracted = node.split("#")[-1].lower()
        if all(word in extracted for word in subj):
            print(f"Subject node: {node}")
            sub_node = node
        if all(word in extracted for word in obj):
            print(f"Object node: {node}")
            obj_node = node

    # more liberal - if no match for full name, try any word in name
    for node in nodes:
        extracted = None  # captures tag of ontology node, capturing its value
        if sub_node is not None and obj_node is not None:
            # if both found, exit loop
            break
        if "fred" not in node:
            extracted = node.split("/")[-1].lower()
        else:
            extracted = node.split("#")[-1].lower()
        if any(word in extracted for word in subj) and sub_node is None:
            print(f"Subject node: {node}")
            sub_node = node
        if any(word in extracted for word in obj) and obj_node is None:
            # Same as above, but for logic
            print(f"Object node: {node}")
            obj_node = node
        if deg_abrs is not None:
            # if "Education", try to match degree abbreviation
            for ab in deg_abrs:
                if re.search(ab, extracted, re.IGNORECASE) and obj_node is None:
                    print(f"Object node: {node}")
                    obj_node = node

    # Final pass, match any words in degree description to an object node - ie 'honorary degree' matches 'degree' or 'honorary doctorate'
    for node in nodes:
        extracted = None
        if sub_node is not None and obj_node is not None:
            # if both found, exit loop
            break
        if "fred" not in node:
            extracted = node.split("/")[-1].lower()
        else:
            extracted = node.split("#")[-1].lower()
        if deg_abrs is not None:
            if any(word in extracted for word in obj) and obj_node is None:
                print(f"Object node: {node}")
                obj_node = node
        else:
            break

    sub_node = "Not Found" if sub_node is None else sub_node
    obj_node = "Not Found" if obj_node is None else obj_node

    return (sub_node, obj_node)


def generate_terminal_node_df(
    snippets: str, rdf_dir: str, out_dir: str, tag: str, existing: str = None
):
    """Generates a dataframe of terminal nodes for each RDF graph provided in rdf_dir

    Args:
        snippets (str): Path to json file containing snippet info
        rdf_dir (str): Path to directory containing RDF files corresponding to snippets
        out_dir (str): Path to directory to store dataframe
        tag (str): A unique tag for identifying this experiment
        existing (str, optional): Path to an existing dataframe to append new dataframe to. Defaults to None.
    """

    now = datetime.datetime.now()
    print("-" * 30)
    print("Beginning generate_terminal_node_df()")
    print("-" * 30)
    print(f"Snippet File: {snippets}")
    print(f"RDF Dir: {rdf_dir}")

    node_dict = dict()

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
        node_dict[uid] = dict()

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
            node_dict[uid]["sub"] = "Not Found"
            node_dict[uid]["obj"] = "Not Found"
            continue

        term_nodes = find_terminal_nodes(rdf_path, subj, obj, db_subj, db_obj)

        node_dict[uid]["sub"] = term_nodes[0]
        node_dict[uid]["obj"] = term_nodes[1]

    df = pd.DataFrame.from_dict(node_dict, orient="index")

    out_file = generate_out_file("terminal_nodes.pkl", out_dir, tag)

    if existing:
        df_existing = pd.read_pickle(existing)
        df = pd.concat([df_existing, df])

    df.to_pickle(out_file)
    print(f"Terminal Nodes written to {out_file}")
    print("Completed generate_terminal_node_df() execution")
    print("-" * 30)

