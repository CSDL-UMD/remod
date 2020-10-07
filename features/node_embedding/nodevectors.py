#!/usr/bin/python3
'''
Run node2vec
'''

import os
import networkx as nx
import pickle
import datetime
import logging
from typing import Type
from nodevectors import Node2Vec
from utils.nx_helpers import uri_to_str
from utils.file import generate_out_file, directory_check

def get_w2vparams(window: int, negative: int = 10, min_count: int = 1, iter: int = 10, batch_words: int = 1, **extras) -> dict:
    """Get w2v paramater dictionary. Ignores extras

    Args:
        window (int): The size of the context window
        negative (int, optional): The number of negative examples to include. Default 10.
        min_count (int, optional): The minimum number of times a node must appear to be included. Default 1.
        iter (int, optional): Number of epochs to run Word2Vec. Default 10
        batch_words (int, optional): How many nodes to read in as batch. Defaults to 1.

    Returns:
        dict: Dictionary of word2vec parameters
    """

    d = {
        'window': window,
        'negative': negative,
        'min_count': min_count,
        'iter': iter,
        'batch_words': batch_words
    }

    return d

def get_n2vparams(dimensions: int, walk_length: int, num_walks: int, p: float, q: float, workers: int, w2vparams: dict) -> dict:
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
        'n_components'   : dimensions,
        'walklen'        : walk_length,
        'epochs'         : num_walks,
        'return_weight'  : p,
        'neighbor_weight': q,
        'threads'        : workers,
        'w2vparams'      : w2vparams
    }

    return d

def nodevec(graph: str, output_dir: str, directed: bool, tag: str, params: dict) -> None:
    

    # Ensure directories exist
    directory_check(output_dir)
    directory_check(output_dir + '/models')
    directory_check(output_dir + '/embeddings')
    temp_dir = output_dir + '/temp'
    directory_check(temp_dir)

    w2vparams = get_w2vparams(**params)
    node2vec_init = get_n2vparams(w2vparams=w2vparams, **params)
    

    print("Beginning node2vec script")
    print("File: %s" % graph)
    for key, value in node2vec_init.items():
        print("%s: %s" %(key, value))
    for key, value in w2vparams.items():
        print("%s: %s" %(key, value))

    G = nx.read_gpickle(graph)
    G = uri_to_str(G)

    if not directed:
        G = G.to_undirected()

    n2v_model = Node2Vec(**node2vec_init)
    n2v_model.fit(G)

    embedding_file = generate_out_file('nodevectors_embeddings.pkl', out_dir + 'embeddings/', tag)
    model_file = generate_out_file('nodevectors_model.pkl', out_dir + 'models/', tag)

    # Save embeddings
    n2v_model.model.wv.save_word2vec_format(embedding_file)
    print("Embeddings saved to %s" % embedding_file)

    # Save model
    n2v_model.model.save(model_file)
    print("Model saved to %s" % embedding_file)

    print("Completed nodevectors.py")

# if __name__ == "__main__":
#     #TODO, build this out with CLI params