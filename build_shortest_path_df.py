"""
Task 5

Generate a dataframe of shortest path calculations
"""

import argparse
import config
import os
import datetime
from features.shortest_path import generate_sp_df
from utils.file import get_experiment_tag, directory_check, generate_out_file
from utils.grec import json_relation_tag


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Calculate shortest paths for GREC subjects/objects"
    )
    now = datetime.datetime.now().strftime("%y%m%d")

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
        default=config.GREC_JSON_DIR,
        help=f"Set filepath to GREC jsons, default {config.GREC_JSON_DIR}",
    )

    parser.add_argument(
        "--rdf-dir",
        "-rdf",
        dest="rdf_dir",
        type=str,
        default=config.GREC_RDF_DIR,
        help=f"Set filepath to RDFs, default {config.GREC_RDF_DIR}",
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
        "--input-tag",
        "-itag",
        dest="in_tag",
        type=str,
        help="The experiment tag for the input N2V model, i.e. model-<tag>.pkl",
    )

    parser.add_argument(
        "--output-tag",
        "-otag",
        dest="out_tag",
        type=str,
        default=now,
        help="Set a unique tag for output files from this experiment, default <weight(if true)>-<dir(if true)>-<input_tag>-YYMMDD",
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    arg_dict = vars(args)
    assert args.in_tag is not None, "Must provide tag for node2vec model"

    directory_check(args.n2v_model_dir, create=False)
    directory_check(args.grec_dir, create=False)
    directory_check(args.rdf_dir, create=False)
    directory_check(args.out_dir)

    n2v_model_file = args.n2v_model_dir + "/model-" + args.in_tag + ".pkl"
    tag = ""

    if args.weighted:
        tag += "weight-"
    if args.directed:
        tag += "dir-"
    itag = get_experiment_tag(n2v_model_file)
    tag += f"{itag}-{args.out_tag}"

    arg_dict.pop("n2v_model_dir")
    arg_dict.pop("in_tag")
    arg_dict.pop("out_tag")

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
                tag=tag,
                weighted=args.weighted,
                directed=args.directed,
                existing=existing_df,
            )

now = datetime.datetime.now()
print(f"Finished creating shortest path dataframe {existing_df}")
print(now)
