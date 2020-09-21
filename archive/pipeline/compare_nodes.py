"""Tasks:
        - match rdf id to raw data
            - find entry
                -get both entity names
        - match entity names to nodes
        - see how far nodes are from one another
            - metrics to log:
                1) RDF ID
                2) Type of Relation
                3) Distance
                4) Ranking ---> to csv
                5) Number of Nodes
                6) Number of Combos
"""
import os
import json
import re
from rdf_to_graph import rdf2Graph
from math import factorial as fac
import ntpath
import csv

def combo_calc(num_obj, sample_size):
    n_fac = fac(num_obj)
    r_fac = fac(sample_size)

    return int(n_fac/(r_fac*fac(num_obj - sample_size)))



instance_types = ["date_of_birth", "education", "institution", "place_of_birth", "place_of_death"]
snippet_dir = snakemake.input[0]
ranking_file = snakemake.input[1]
rdf_file = snakemake.input[2]
log_file = snakemake.log[0]

id = ntpath.basename(rdf_file).replace(".rdf", "")
instance_type = None
for i_type in instance_types:
    if i_type in rdf_file:
        instance_type = i_type
        break


sub = ""
obj = ""
rdf = rdf2Graph(rdf_file)


# get json file path
snippet_files = os.listdir(snippet_dir)
snippet_json = ""
for s_file in snippet_files:
    if instance_type in s_file:
        snippet_json = snippet_dir + s_file

with open(snippet_json, 'r') as f:
    relations = json.loads(f.read())

    # find object and subject corresponding to id from snippet json
    for relation in relations:
        if id in relation['UID']:
            sub = relation['sub'].split()[-1]
            if "education" not in instance_type:
                obj = relation['obj'].split()[0]
            else:
                obj = re.search('(\d{4})', relation['obj']).group(1) # only grabbing year for search
            break

rank, dist = -1, -1
# get subj and obj num
with open(ranking_file, 'r') as f:
    for line in f.read().split('#&\n')[1:]:
        if re.search(sub, line, re.IGNORECASE) and re.search(obj, line, re.IGNORECASE):
            dist = float(line[3:11])
            rank = int(re.search('(^\d+)', line).group(1))

final_analysis = {
    'id': id,
    'relation': instance_type,
    'subject': sub,
    'object': obj,
    'similarity': dist,
    'rank': rank,
    'num_nodes': rdf.num_nodes,
    'num_combos': combo_calc(rdf.num_nodes, 2)
    
}


csv_file = "../../../all_data/relation_extraction/data/rdf_processing/results.csv"
new_file = False
if os.stat(csv_file).st_size == 0:
    new_file = True

with open(csv_file, 'a') as f:
    w = csv.DictWriter(f, final_analysis.keys())
    if new_file is True:
        w.writeheader()
    w.writerow(final_analysis)


with open(log_file, 'w') as f:
    for i in final_analysis:
        f.write("%s: %s" % (i, str(final_analysis[i])))