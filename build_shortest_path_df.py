"""
Task 6

Generate a dataframe of shortest path calculations
"""

import argparse
import config
import os
import datetime
from features import sp_params
from features.shortest_path import generate_sp_df
from utils.file import get_experiment_tag, directory_check, generate_out_file
from utils.grec import json_relation_tag


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Calculate shortest paths for GREC subjects/objects"
    )

    parser.add_argument(
        "--weighted",
        dest="weighted",
        action="store_true",
        default=False,
        help="Weight the edges, and compute Dijkstra's shortest path",
    )

    parser.add_argument(
        "--directed",
        dest="directed",
        action="store_true",
        default=False,
        help="Use Directed Graph for shortest path calculations",
    )

    parser.add_argument(
        "--n2v-model-dir",
        "-n2v",
        dest="n2v_model_dir",
        type=str,
        default=config.N2V_MODEL,
        help=f"Set directory to Node2Vec Model (.pkl), default {config.N2V_MODEL}",
    )

    parser.add_argument(
        "--grec-dir",
        "-grec",
        dest="grec_dir",
        type=str,
        default=config.JSON_DIR,
        help=f"Set filepath to GREC jsons, default {config.JSON_DIR}",
    )

    parser.add_argument(
        "--rdf-dir",
        "-rdf",
        dest="rdf_dir",
        type=str,
        default=config.RDF_DIR,
        help=f"Set filepath to RDFs, default {config.RDF_DIR}",
    )

    parser.add_argument(
        "--out-dir",
        "-out",
        dest="out_dir",
        type=str,
        default=config.SP_DIR,
        help=f"Set filepath for output dataframe, default {config.SP_DIR}",
    )

    parser.add_argument(
        "--node-file",
        dest="node_file",
        type=str,
        default=(config.SP_NODES + '/terminal_nodes.pkl'),
        help=f"Set filepath for terminal node dataframe, default {(config.SP_NODES + '/terminal_nodes.pkl')}",
    )

    parser.add_argument(
        "--custom",
        dest="custom",
        type=str,
        default="None",
        help="Name of features/sp_params dictionary entry. If nothing entered, defaults to best models"
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%y%m%d")

    args = arg_parse()
    arg_dict = vars(args)

    directory_check(args.n2v_model_dir, create=False)
    directory_check(args.grec_dir, create=False)
    directory_check(args.rdf_dir, create=False)
    directory_check(args.out_dir)

    n2v_model_files = []
    tags = None

    if args.custom != "None":
        tags = [sp_params.sp_param_dict[args.custom]]
    else:
        tags = [sp_params.sp_param_dict["best"]]
    
    models = os.listdir(args.n2v_model_dir)

    for tag_i in tags:
        n2v_model_file = None
        for model in models:
            if tag_i in model and model.endswith('.pkl'):
                n2v_model_file = args.n2v_model_dir + '/' + model
                n2v_model_files.append(n2v_model_file)
                break

        tag = ""

        if args.weighted:
            tag += "weight-"
        if args.directed:
            tag += "dir-"
        tag += f"{tag_i}-{now}"

        arg_dict.pop("n2v_model_dir")
        arg_dict.pop("custom")

        now = datetime.datetime.now()
        print("build_shortest_path_df.py")
        print("-" * 30)
        print(f"Now: {now}")
        print(f"N2V Model: {n2v_model_file}")
        print(f"JSON Dir: {args.grec_dir}")
        print(f"RDF Dir: {args.rdf_dir}")
        print(f"Output Dir: {args.out_dir}")
        print(f"Weighted?: {args.weighted}")
        print(f"Directed?: {args.directed}")
        print(f"Experiment tag: {tag}")

        rdf_sub_dirs = [str(args.rdf_dir + "/" + x) for x in os.listdir(args.rdf_dir)]
        jsons = [str(args.grec_dir + "/" + x) for x in os.listdir(args.grec_dir)]

        pairs = []
        for json in jsons:
            for rdf_sub_dir in rdf_sub_dirs:
                if json_relation_tag(json) in rdf_sub_dir:
                    pairs.append([json, rdf_sub_dir])
                    continue

        for i, pair in enumerate(pairs):
            json, rdf_dir = pair
            if i == 0:
                generate_sp_df(
                    n2v_model_file=n2v_model_file,
                    snippets=json,
                    rdf_dir=rdf_dir,
                    out_dir=args.out_dir,
                    node_file=args.node_file,
                    tag=tag,
                    weighted=args.weighted,
                    directed=args.directed,
                )
            else:
                existing_df = generate_out_file("sp_df.pkl", args.out_dir, tag)
                generate_sp_df(
                    n2v_model_file=n2v_model_file,
                    snippets=json,
                    rdf_dir=rdf_dir,
                    out_dir=args.out_dir,
                    node_file=args.node_file,
                    tag=tag,
                    weighted=args.weighted,
                    directed=args.directed,
                    existing=existing_df,
                )

now = datetime.datetime.now()
print(f"Finished creating shortest path dataframe {existing_df}")
print(now)
