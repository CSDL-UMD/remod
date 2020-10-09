#!/usr/bin/python3
"""
Run node2vec
"""

import os
import networkx as nx
import pickle
import datetime
import logging
import argparse
from typing import Type
from nodevectors import Node2Vec
import sys

sys.path.append("RESOGE/src")
from utils.nx_helpers import uri_to_str
from utils.file import generate_out_file, directory_check


def get_w2vparams(
    window: int,
    negative: int = 5,
    min_count: int = 1,
    iter: int = 5,
    batch_words: int = 10000,
    **extras,
) -> dict:
    """Get w2v paramater dictionary. Ignores extras

    Args:
        window (int): The size of the context window
        negative (int, optional): The number of negative examples to include. Default 5.
        min_count (int, optional): The minimum number of times a node must appear to be included. Default 1.
        iter (int, optional): Number of epochs to run Word2Vec. Default 5
        batch_words (int, optional): How many nodes to read in as batch. Defaults to 10000.

    Returns:
        dict: Dictionary of word2vec parameters
    """

    d = {
        "window": window,
        "negative": negative,
        "min_count": min_count,
        "iter": iter,
        "batch_words": batch_words,
    }

    return d


def get_n2vparams(
    dimensions: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    workers: int,
    w2vparams: dict,
    **extras,
) -> dict:
    """Generates dictionary of Nodevector Parameters

    Args:
        dimensions (int): Size of vectors to generate
        walk_length (int): Length of random walks
        num_walks (int): Number of walks to generate per node
        p (float): The return parameter (likelyhood of returning to previously visited node)
        q (float): The In-Out parameter (likelyhood of visiting unexplored node)
        workers (int): Number of workers
        w2vparams (dict): Dictionary of w2v parameters

    Returns:
        dict: Dictionary of Node2Vec/Nodevector parameters
    """

    d = {
        "n_components": dimensions,
        "walklen": walk_length,
        "epochs": num_walks,
        "return_weight": p,
        "neighbor_weight": q,
        "threads": workers,
        "w2vparams": w2vparams,
    }

    return d


def nodevec(
    graph: str, output_dir: str, directed: bool, tag: str, params: dict
) -> None:

    # Ensure directories exist
    directory_check(output_dir)
    directory_check(output_dir + "/models")
    directory_check(output_dir + "/embeddings")
    temp_dir = output_dir + "/temp"
    directory_check(temp_dir)

    w2vparams = get_w2vparams(**params)
    node2vec_init = get_n2vparams(w2vparams=w2vparams, **params)

    print("Beginning node2vec script")
    print("File: %s" % graph)
    for key, value in node2vec_init.items():
        print("%s: %s" % (key, value))
    for key, value in w2vparams.items():
        print("%s: %s" % (key, value))

    G = nx.read_gpickle(graph)
    G = uri_to_str(G)

    if not directed:
        G = G.to_undirected()

    n2v_model = Node2Vec(**node2vec_init)
    n2v_model.fit(G)

    embedding_file = generate_out_file("embeddings.pkl", out_dir + "embeddings/", tag)
    model_file = generate_out_file("model.pkl", out_dir + "models/", tag)

    # Save embeddings
    n2v_model.model.wv.save_word2vec_format(embedding_file)
    print("Embeddings saved to %s" % embedding_file)

    # Save model
    n2v_model.model.save(model_file)
    print("Model saved to %s" % embedding_file)

    print("Completed nodevectors.py")


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Run Node2Vec on a given pickled graph"
    )
    now = datetime.datetime.now().strftime("%y%m%d")

    parser.add_argument(
        "--dimensions",
        "-d",
        dest="dimensions",
        type=int,
        default=256,
        help="The number of dimensions for embedded vectors, default 256",
    )

    parser.add_argument(
        "--walk-length",
        "-wl",
        dest="walk_length",
        type=int,
        default=50,
        help="The length of randomly generated walks, default 50",
    )

    parser.add_argument(
        "--num-walks",
        "-nw",
        dest="num_walks",
        type=int,
        default=200,
        help="The number of randomly generated walks per node, default 200",
    )

    parser.add_argument(
        "--return",
        "-p",
        dest="p",
        type=float,
        default=1,
        help="The return parameter - likelyhood to return to already visited node when generating walks, default 1",
    )

    parser.add_argument(
        "--in-out",
        "-q",
        dest="q",
        type=float,
        default=1,
        help="The in-out parameter - likelyhood to explore an unvisited node when generating walks, default 1",
    )

    parser.add_argument(
        "--workers",
        "-w",
        dest="workers",
        type=int,
        default=4,
        help="The number of cores to use, default 4",
    )

    parser.add_argument(
        "--window",
        "-win",
        dest="window",
        type=int,
        default=15,
        help="The size of the skipgram window, default 15",
    )

    parser.add_argument(
        "--min-count",
        "-mc",
        dest="min_count",
        type=int,
        default=1,
        help="The max frequency of words to be included, default 1",
    )

    parser.add_argument(
        "--directed",
        "-di",
        dest="directed",
        action="store_true",
        default=False,
        help="Process the graph as an directed graph, default False",
    )

    parser.add_argument(
        "--graph_file", "-graph", dest="graph_file", type=str, help=f"Set input graph",
    )

    parser.add_argument(
        "--out-dir", "-out", dest="out_dir", type=str, help=f"Set output directory",
    )

    parser.add_argument(
        "--output-tag",
        "-otag",
        dest="out_tag",
        type=str,
        default=now,
        help="Set a unique tag for output files from this experiment, default YYMMDD",
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    arg_dict = vars(args)
    assert args.graph_file is not None, "Must provide filepath for corpus graph"
    assert args.out_dir is not None, "Must provide output directory for corpus graph"

    now = datetime.datetime.now().strftime("%y%m%d")

    tag = arg_dict.pop("out_tag")
    tag += f"-{now}"
    out_dir = arg_dict.pop("out_dir")
    directed = arg_dict.pop("directed")
    in_graph = arg_dict.pop("graph_file")

    print(f"{datetime.datetime.now()}")
    print(f"Beginning NodeVectors")
    print("-" * 30)
    print(f"In Graph: {in_graph}")
    print(f"Output Dir: {out_dir}")
    print(f"Unique Tag: {tag}")

    nodevec(in_graph, out_dir, directed, tag, arg_dict)

    print(f"Finished NodeVectors")
    print(f"{datetime.datetime.now()}")
    print("-" * 30)
