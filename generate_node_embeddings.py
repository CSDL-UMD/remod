'''
Task 4

Generate embeddings for corpus graph nodes
'''

import argparse

def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(description="Run Node2Vec on a given pickled graph")
    now = datetime.datetime.now().strftime("%b-%d-%y")

    parser.add_argument(
        '--nodevectors',
        dest='nodevectors',
        action='store_true',
        default=False,
        help="Uses the nodevectors package, rather than SNAP Node2Vec"
    )

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
