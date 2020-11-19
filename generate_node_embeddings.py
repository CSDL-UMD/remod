"""
Task 5

Generate embeddings for corpus graph nodes
"""

import argparse
import datetime

import config
from features.node_embedding import n2v, nodevec, n2v_params
from utils.file import directory_check


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Run Node2Vec on a given pickled graph"
    )

    parser.add_argument(
        "--nodevectors",
        dest="nodevectors",
        action="store_true",
        default=False,
        help="Uses the nodevectors package, rather than SNAP Node2Vec",
    )

    parser.add_argument(
        "--custom",
        dest="custom",
        type=str,
        default="None",
        help="Name of node_embeddings/n2v_params dictionary. If nothing entered, defaults to best models",
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
        "--graph_dir",
        "-graph",
        dest="graph_dir",
        type=str,
        default=config.GRAPH_DIR,
        help=f"Set input graph directory, default {config.GRAPH_DIR}",
    )

    parser.add_argument(
        "--out-dir",
        "-out",
        dest="out_dir",
        type=str,
        default=config.N2V_DIR,
        help=f"Set output directory, default {config.N2V_DIR}",
    )

    parser.add_argument(
        "--input-tag",
        "-itag",
        dest="in_tag",
        type=str,
        help="The experiment tag for the input corpus graph.",
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%y%m%d")

    args = arg_parse()
    arg_dict = vars(args)
    assert args.in_tag is not None, "Must provide tag for corpus graph"

    directory_check(args.graph_dir, create=False)
    in_graph = args.graph_dir + "/corpus_graph-" + args.in_tag + ".pkl"
    itag = args.in_tag.split("-")[0]
    tag = None

    nv = arg_dict["nodevectors"]

    params = None

    out_dir = arg_dict.pop("out_dir")
    directed = arg_dict.pop("directed")

    if args.custom != "None":
        params = [n2v_params.n2v_param_dict[args.custom]]
    else:
        params = [n2v_params.n2v_param_dict["best"]]

    for run in params:

        if nv:
            tag = f"{itag}-nv-d{run['dimensions']}-wl{run['walk_length']}-nw{run['num_walks']}-win{run['window']}-p{run['p']}-q{run['q']}-{now}"
        else:
            tag = f"{itag}-d{run['dimensions']}-wl{run['walk_length']}-nw{run['num_walks']}-win{run['window']}-p{run['p']}-q{run['q']}-{now}"
        
        if nv:
            nodevectors.nodevec(in_graph, out_dir, directed, tag, run)
            print(f"Finished generating node embeddings for {in_graph}")
        else:
            n2v.n2v(in_graph, out_dir, directed, tag, run)
            print(f"Finished generating node embeddings for {in_graph}")

