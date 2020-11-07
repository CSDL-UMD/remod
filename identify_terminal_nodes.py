"""
Task 3

Identify the terminal nodes for each relation
"""

import argparse
import config
import os
import datetime
from preproc.terminal_nodes import generate_terminal_node_df
from utils.file import get_experiment_tag, directory_check, generate_out_file
from utils.grec import json_relation_tag


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Find Terminal Nodes from RDFs GREC subjects/objects"
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
        default=config.SP_NODES,
        help=f"Set filepath for output dataframe, default {config.SP_NODES}",
    )

    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="None",
        help="Experimental tag. If none, defaults to YYMMDD",
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%y%m%d")

    args = arg_parse()
    arg_dict = vars(args)

    directory_check(args.grec_dir, create=False)
    directory_check(args.rdf_dir, create=False)
    directory_check(args.out_dir)

    tag = args.tag if args.tag != "None" else now

    arg_dict.pop("tag")

    now = datetime.datetime.now()
    print("identify_terminal_nodes.py")
    print("-" * 30)
    print(f"Now: {now}")
    print(f"JSON Dir: {args.grec_dir}")
    print(f"RDF Dir: {args.rdf_dir}")
    print(f"Output Dir: {args.out_dir}")

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
            generate_terminal_node_df(
                snippets=json, rdf_dir=rdf_dir, out_dir=args.out_dir, tag=tag
            )
        else:
            existing_df = generate_out_file("terminal_nodes.pkl", args.out_dir, tag)
            generate_terminal_node_df(
                snippets=json,
                rdf_dir=rdf_dir,
                out_dir=args.out_dir,
                tag=tag,
                existing=existing_df
            )

now = datetime.datetime.now()
print(f"Finished creating shortest path dataframe {existing_df}")
print(now)
