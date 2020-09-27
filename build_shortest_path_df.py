'''
Task 5

Generate a dataframe of shortest path calculations
'''

import argparse
from features.shortest_path import generate_sp_df

def arg_parse(arg_list=None):
    parser = argparse.ArgumentParser(description="Calculate shortest paths for GREC subjects/objects")
    now = datetime.datetime.now().strftime("%y%m%d")

    parser.add_argument(
        '--weighted',
        dest='dijkstra',
        action='store_true',
        default=False,
        help="Weight the edges, and compute Dijkstra's shortest path"
    )

    parser.add_argument(
        '--directed',
        dest='directed',
        action='store_true',
        default=False,
        help="Use Directed Graph for shortest path calculations"
    )

    parser.add_argument(
        '--n2v-model',
        '-n2v',
        dest="n2v_model",
        type=str,
        default='/data/mjsumpter/nodevectors_ontology_100wl_100nw/models/model-ontology_stitch-100wl-100nw.pkl',
        help="Set filepath to local Node2Vec Model (.pkl), default /data/mjsumpter/nodevectors_ontology_100wl_100nw/models/model-ontology_stitch-100wl-100nw.pkl"
    )

    parser.add_argument(
        '--grec-dir',
        '-grec',
        dest="grec_dir",
        type=str,
        default='/data/mjsumpter/grec/',
        help="Set filepath to GREC jsons, default /data/mjsumpter/grec/"
    )

    parser.add_argument(
        '--rdf-dir',
        '-rdf',
        dest="rdf_dir",
        type=str,
        default='/data/mjsumpter/rdfs/',
        help="Set filepath to GREC jsons, default /data/mjsumpter/rdfs/"
    )

    parser.add_argument(
        '--out-dir',
        '-out',
        dest="out_dir",
        type=str,
        default='/data/mjsumpter/to_be_sorted/',
        help="Set filepath for output dataframe, default /data/mjsumpter/to_be_sorted/"
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
