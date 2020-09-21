#!/usr/bin/python3
'''
Run node2vec
'''

import os
import networkx as nx
import pickle
import datetime
import logging
import argparse
from node2vec import Node2Vec

def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(description="Run Node2Vec on a given pickled graph")
    now = datetime.datetime.now().strftime("%b-%d-%y")

    parser.add_argument(
        '--dimensions',
        '-d',
        dest='dims',
        type=int,
        default=128,
        help="The number of dimensions for embedded vectors, default 128"
    )

    parser.add_argument(
        '--walk-length',
        '-wl',
        dest='walk_length',
        type=int,
        default=50,
        help="The length of randomly generated walks, default 50"
    )

    parser.add_argument(
        '--num-walks',
        '-nw',
        dest='num_walks',
        type=int,
        default=50,
        help="The number of randomly generated walks per node, default 50"
    )

    parser.add_argument(
        '--return',
        '-p',
        dest='p_return',
        type=float,
        default=1,
        help="The return parameter - likelyhood to return to already visited node when generating walks, default 1"
    )

    parser.add_argument(
        '--in-out',
        '-q',
        dest='q_in_out',
        type=float,
        default=1,
        help="The in-out parameter - likelyhood to explore an unvisited node when generating walks, default 1"
    )

    parser.add_argument(
        '--workers',
        '-w',
        dest='workers',
        type=int,
        default=4,
        help="The number of cores to use, default 4"
    )

    parser.add_argument(
        '--window',
        '-win',
        dest='window',
        type=int,
        default=25,
        help="The size of the skipgram window, default 25"
    )

    parser.add_argument(
        '--min-count',
        '-mc',
        dest='min_count',
        type=int,
        default=1,
        help="The max frequency of words to be included, default 1"
    )

    parser.add_argument(
        '--undirected',
        '-undir',
        dest='undirected',
        action='store_true',
        default=False,
        help="Process the graph as an undirected graph, default False"
    )

    parser.add_argument(
        '--in-graph',
        '-in',
        dest='in_graph',
        type=str,
        default=None,
        help="Set input graph, default None"
    )

    parser.add_argument(
        '--out-dir',
        '-out',
        dest='out_dir',
        type=str,
        default=None,
        help="Set output directory, default None"
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

    in_graph = args.in_graph
    out_dir = args.out_dir
    exp_tag = args.exp_tag
    undirected = args.undirected

    node2vec_init = {
        'dimensions'   : args.dims,
        'walk_length'  : args.walk_length,   # max nodes in a relation * 2 = 200
        'num_walks'    : args.num_walks,
        'p'            : args.p_return,   # lower to encourage graph exploration
        'q'            : args.q_in_out,
        'workers'      : args.workers,
        'temp_folder'  : out_dir + 'temp/'
    }

    node2vec_fit = {
        'window'      : args.window,    # half of max nodes
        'min_count'   : args.min_count,
        # 'batch_words' : 4
    }

    ############

    logging.info("Beginning node2vec script")
    logging.info("File: %s" % in_graph)
    for key, value in node2vec_init.items():
        logging.info("%s: %s" %(key, value))
    for key, value in node2vec_fit.items():
        logging.info("%s: %s" %(key, value))

    G = nx.read_gpickle(in_graph)

    if undirected:
        G = G.to_undirected()

    try:
        node2vec=Node2Vec(G, **node2vec_init)
        model = node2vec.fit(**node2vec_fit)
    except Exception as e:
        logging.error("Failed to run Node2Vec on Graph")
        logging.error(e.__doc__)

    embedding_file = generate_out_file('embeddings.pkl', out_dir + 'embeddings/', exp_tag)
    model_file = generate_out_file('model.pkl', out_dir + 'models/', exp_tag)

    # Save embeddings
    model.wv.save_word2vec_format(embedding_file)
    logging.info("Embeddings saved to %s" % embedding_file)

    # Save model
    model.save(model_file)
    logging.info("Model saved to %s" % embedding_file)

    logging.info("Completed Node2Vec.py")

if __name__ == "__main__":
    main()