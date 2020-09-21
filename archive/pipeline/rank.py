import numpy as np
from nodemap import nodemap as nm
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

def similarity_transform(node_string):
    return (int(node_string.strip('()').split(',')[0]), int(node_string.strip('()').split(',')[1]))

'''
read each vector in .emd into python
iterate through every combo
    calculate similarity
    store both nodes, and similarity between them
sort similaritys from smallest to largest
read in nodemap
    change nodes to node content
'''

nodevec_file = snakemake.input[0]
nodemap_file = snakemake.input[1]
outfile = snakemake.output[0]

# nodevec_file = "../examples/test.emd"
# nodemap = "../examples/test.edgemap"
# outfile = "../examples/test.ranking"

nodemap = nm(nodemap_file,nodevec_file)

# gets all possible combinations of nodes
node_combos = tuple(combinations(tuple(i for i,_ in enumerate(nodemap.numpy_vectors)), 2))

# fill dictionary with node combinations, and their similaritys
similarity_dict = dict()
for combo in node_combos:
    dist = cosine_similarity(nodemap.numpy_vectors[combo[0]].reshape(1,-1), nodemap.numpy_vectors[combo[1]].reshape(1,-1))[0][0]
    similarity_dict[str(combo)] = dist

# sorted similaritys
dist_vals = sorted((value, key) for (key, value) in similarity_dict.items())

# transpose vector number to vector content
dist_vals = tuple( (pair[0], similarity_transform(pair[1]) ) for pair in dist_vals) # transform node string to node ints

# write ranking to file
with open(outfile, 'w') as f:
    f.write("Rank from smallest to biggest:#&\n")
    for idx, item in enumerate(dist_vals):
        f.write("%d\t\t%f\t\t(%d,%d)\t\t(%s,%s)#&\n" % (idx+1, item[0], item[1][0], item[1][1], nodemap[item[1][0]], nodemap[item[1][1]]) )


