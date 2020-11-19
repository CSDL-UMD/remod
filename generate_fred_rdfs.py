"""
Task 2

Read in all snippets from GREC and ClaimReview. Calculate average number of words and collect snippets within 1 std.dev.

Pass snippets to FRED to generate RDF graphs
"""

import argparse
import os
from utils.file import absolute_paths, directory_check
from utils.grec import get_metrics, get_snippet, json_relation_tag
from utils.api import get_api_key
from preproc.fred.fred_extraction import generate_rdfs
import config
import json


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Generate FRED RDFs from GREC snippets"
    )
    parser.add_argument(
        "--api-method",
        "-api",
        dest="api",
        help="The method for API calls, default 'limited'",
        type=str,
        default="limited",
    )
    parser.add_argument(
        "--api-key",
        "-key",
        dest="api_key",
        help=f"The filepath for the API key, default {config.FRED_LMTD}",
        type=str,
        default=config.FRED_LMTD,
    )
    parser.add_argument(
        "--grec-dir",
        "-grec",
        dest="grec",
        help=f"The directory path for the GREC, default {config.JSON_DIR}",
        type=str,
        default=config.JSON_DIR,
    )
    parser.add_argument(
        "--rdf-dir",
        "-rdf",
        dest="rdf",
        help=f"The directory to save the RDFs to, default {config.RDF_DIR}",
        type=str,
        default=config.RDF_DIR,
    )
    # Parses and returns args
    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


def main(api_method, api_key_file, grec_dir, rdf_dir):

    grec_files = absolute_paths(grec_dir)

    for j_file in grec_files:

        # get metrics for excluding outlier snippets
        metrics = get_metrics(j_file)
        max = metrics["max_no_outlier"]
        min = metrics["min_no_outlier"]
        relation = json_relation_tag(j_file)

        # Create relation-labelled directory for storing rdfs
        out_dir = rdf_dir + "/" + relation + "/"

        print(f"\nBeginning parsing GREC file: {j_file}\n")
        print(f"Excluding snippets with over {max} words, and under {min} words")

        # Create dictionary of UIDS: snippets
        relations = None
        #   Read json
        with open(j_file) as f:
            relations = json.loads(f.read())
        #   Load dictionary
        snips = dict()
        for relation in relations:
            uid = relation["UID"]
            snippet = get_snippet(relation)
            len_snip = len(snippet.split())
            if (len_snip > max or len_snip < min) and (
                "_cr" not in uid
            ):  # if number of words in snippet is outlier, do not include
                continue
            else:
                snips[uid] = snippet

        generate_rdfs(api_key_file, snips, out_dir, api_method)


if __name__ == "__main__":
    args = arg_parse()

    api_method = args.api
    api_key_file = args.api_key
    grec_dir = args.grec
    rdf_dir = args.rdf

    directory_check(rdf_dir)

    main(api_method, api_key_file, grec_dir, rdf_dir)
