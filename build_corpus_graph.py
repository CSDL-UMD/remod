"""
Task 4

Takes all rdf graphs and combines them to generate one giant "corpus" graph
"""

import config
import argparse
import datetime
import os
from utils.rdf import get_relations
from data.corpus_graph_builder import create_graph
from utils.file import generate_out_file


def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(description="Stitch RDF graphs")
    now = datetime.datetime.now().strftime("%y%m%d")
    parser.add_argument(
        "--leave-fred-nodes",
        "-l",
        dest="fred",
        action="store_true",
        default=False,
        help="Don't Collapse FRED nodes that point to DBPedia or VBNet Nodes, default False",
    )

    parser.add_argument(
        "--append-rdf-ids",
        "-aid",
        dest="append_rdf_id",
        action="store_true",
        default=False,
        help="Append all RDF UID's to FRED Nodes. Useful for keeping subgraphs separate. Default False",
    )

    parser.add_argument(
        "--in-dir",
        "-in",
        dest="rdf_dir",
        type=str,
        default=config.GREC_RDF_DIR,
        help=f"Set input directory of RDFs, default {config.GREC_RDF_DIR}",
    )

    parser.add_argument(
        "--out-dir",
        "-out",
        dest="out_dir",
        type=str,
        default=config.GREC_GRAPH_DIR,
        help=f"Set output directory, default {config.GREC_GRAPH_DIR}",
    )

    parser.add_argument(
        "--experiment-tag",
        "-tag",
        dest="exp_tag",
        type=str,
        default=now,
        help="Set a unique tag for output files from this experiment, default <full/ontol>-YYMMDD",
    )

    if arg_list:
        return parser.parse_args(args=arg_list)
    else:
        return parser.parse_args()


def main(fred: bool, append_ids: bool, rdf_dir: str, out_dir: str, tag: str) -> None:
    """
    rdf_dir: Directory where RDFs are located
    out_dir: Directory to output graph and log
    fred: If True, leave FRED nodes intact
    append_ids: If True, append file ids to FRED nodes
    tag: a unique tag for the output files. Defaults to current time
    """
    tag_addend = "ontol" if append_ids else "full"
    tag = tag_addend + "-" + tag

    now = datetime.datetime.now()
    print("build_corpus_graph.py")
    print("----------------------")
    print(f"Now: {now}")
    print(f"RDF Dir: {rdf_dir}")
    print(f"Output Dir: {out_dir}")
    print(f"Keep Fred Nodes?: {fred}")
    print(f"Stitch only on non-FRED nodes?: {append_ids}")
    print(f"Experiment tag: {tag}")

    rdf_sub_dirs = [str(rdf_dir + "/" + x + "/") for x in os.listdir(rdf_dir)]
    for i, entry in enumerate(rdf_sub_dirs):
        print(f"Appending subgraphs from {entry}")
        if i == 0:
            create_graph(entry, out_dir, fred, append_ids, tag)
        else:
            existing_graph = generate_out_file("corpus_graph.pkl", out_dir, tag)
            create_graph(entry, out_dir, fred, append_ids, tag, existing=existing_graph)

    now = datetime.datetime.now()
    print(f"Finished creating corpus graph {existing_graph}")
    print(now)


if __name__ == "__main__":
    args = arg_parse()
    fred = args.fred
    append_ids = args.append_rdf_id
    rdf_dir = args.rdf_dir
    out_dir = args.out_dir
    tag = args.exp_tag

    main(fred, append_ids, rdf_dir, out_dir, tag)
