'''
Task 4

Generate embeddings for corpus graph nodes
'''

import argparse
import datetime
from src.features.node_embedding import n2v, nodevectors

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
        dest='dimensions',
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
        dest='p',
        type=float,
        default=1,
        help="The return parameter - likelyhood to return to already visited node when generating walks, default 1"
    )

    parser.add_argument(
        '--in-out',
        '-q',
        dest='q',
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
        '--directed',
        '-di',
        dest='directed',
        action='store_true',
        default=False,
        help="Process the graph as an directed graph, default False"
    )

    parser.add_argument(
        '--in-graph',
        '-in',
        dest='in_graph',
        type=str,
        default=config.CORPUS_GRAPH_SM,
        help=f"Set input graph, default {config.CORPUS_GRAPH_SM}"
    )

    parser.add_argument(
        '--out-dir',
        '-out',
        dest='out_dir',
        type=str,
        default=config.N2V_DIR,
        help=f"Set output directory, default {config.N2V_DIR}"
    )

    parser.add_argument(
        '--experiment-tag',
        '-tag',
        dest="exp_tag",
        type=str,
        default=now,
        help="Set a unique tag for output files from this experiment, default d<dims>-wl<wl>-nw<nw>-win<win>-p<p>-q<q>-MM-DD-YY"
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    tag = f"d{args.dimensions}-wl{args.walk_length}-nw{args.num_walks}-win{args.window}-p{args.p}-q{args.q}-{args.exp_tag}"

    arg_dict = vars(args)
    arg_dict.pop('nodevectors')
    tag = arg_dict.pop('exp_tag')
    in_graph = arg_dict.pop('in_graph')
    out_dir = arg_dict.pop('out_dir')
    directed = arg_dict.pop('directed')

    if args.nodevectors:
        # RUN NODEVECTORS
    else:
        n2v.n2v(in_graph, out_dir, directed, tag, arg_dict)
        print(f"Finished generating node embeddings for {in_graph}")