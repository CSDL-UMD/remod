"""
Run SNAP implementation of Node2Vec
"""

import os
import networkx as nx
import pickle
import argparse
import datetime
import logging
from typing import Type
from node2vec import Node2Vec
from utils.file import directory_check, generate_out_file


def n2v_init(
    dimensions: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    workers: int,
    temp_folder: str,
    **extras,
) -> dict:
    """Generates dictionary for initializing Node2Vec Model

    Args:
        dimensions (int): Number of dimensions for embedding
        walk_length (int): Length of random walks to be generated
        num_walks (int): Number of walks to generate per node
        p (float): The return parameter (likelyhood of returning to previously visited node)
        q (float): The In-Out parameter (likelyhood of visiting unexplored node)
        workers (int): Number of workers
        temp_folder (str): Folder for storing temporary checkpoints
        **extras (dict): Extra keywords (to be ignored)

    Returns:
        dict: Dictionary that can be passed to Node2Vec
    """

    d = {
        "dimensions": dimensions,
        "walk_length": walk_length,
        "num_walks": num_walks,
        "p": p,
        "q": q,
        "workers": workers,
        "temp_folder": temp_folder,
    }

    return d


def n2v_fit(window: int, min_count: int = 1, batch_words: int = 1, **extras) -> dict:
    """Returns parameters for Word2Vec fitting of Node2Vec Model

    Args:
        window (int): Size of context window
        min_count (int, optional): How many times a node must be present to be included. Defaults to 1.
        batch_words (int, optional): How many nodes to read in as batch. Defaults to 1.

    Returns:
        dict: A dictionary for passing to node2vec.fit()
    """

    d = {"window": window, "min_count": min_count, "batch_words": batch_words}

    return d


def n2v(graph: str, output_dir: str, directed: bool, tag: str, params: dict) -> None:
    """Runs the SNAP implementation of Node2Vec on a NetworkX graph

    Args:
        graph (str): Path to a pickled NetworkX Graph
        output_dir (str): The directory that will save Node2Vec Model.
        directed (bool): If True, process as directed graph
        tag (str): The tag that will be appended to output files, useful for IDing 
        params (dict): Dictionary of Node2Vec/Word2Vec Parameters
    """

    # Ensure directories exist
    directory_check(output_dir)
    directory_check(output_dir + "/models")
    directory_check(output_dir + "/embeddings")
    temp_dir = output_dir + "/temp"
    directory_check(temp_dir)

    node2vec_init = n2v_init(temp_folder=temp_dir, **params)
    node2vec_fit = n2v_fit(**params)

    print("Beginning node2vec script")
    print("Graph: %s" % graph)
    for key, value in node2vec_init.items():
        print("%s: %s" % (key, value))
    for key, value in node2vec_fit.items():
        print("%s: %s" % (key, value))

    G = nx.read_gpickle(graph)

    if not directed:
        G = G.to_undirected()

    try:
        node2vec = Node2Vec(G, **node2vec_init)
        model = node2vec.fit(**node2vec_fit)
    except Exception as e:
        logging.error("Failed to run Node2Vec on Graph")
        logging.error(e.__doc__)

    embedding_file = generate_out_file(
        "embeddings.pkl", output_dir + "/embeddings", tag
    )
    model_file = generate_out_file("model.pkl", output_dir + "/models", tag)

    # Save embeddings
    model.wv.save_word2vec_format(embedding_file)
    print("Embeddings saved to %s" % embedding_file)

    # Save model
    model.save(model_file)
    print("Model saved to %s" % model_file)

    print("Completed n2v.py")


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

    arg_dict.pop("graph_file")
    tag = arg_dict.pop("out_tag")
    tag += f"-{now}"
    out_dir = arg_dict.pop("out_dir")
    directed = arg_dict.pop("directed")
    in_graph = arg_dict.pop("graph_file")

    print(f"{datetime.datetime.now()}")
    print(f"Beginning Node2Vec")
    print("-" * 30)
    print(f"In Graph: {args.graph_file}")
    print(f"Output Dir: {args.out_dir}")
    print(f"Unique Tag: {tag}")

    n2v(in_graph, out_dir, directed, tag, arg_dict)

    print(f"Finished Node2Vec")
    print(f"{datetime.datetime.now()}")
    print("-" * 30)
