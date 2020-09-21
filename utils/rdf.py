import rdflib
from .file import get_filename
from collections import Counter
import os

def get_relations(rdf_dir: str) -> list:
    '''
    Given the RDF directory, returns a list of folder headings, which should represent relations
    '''
    return [x for x in os.listdir(rdf_dir)]

def get_rdfGraph(rdf_path: str) -> rdflib.Graph:
    g = rdflib.Graph()
    return g.parse(rdf_path)

def graph_meta(rdf: rdflib.Graph = None, rdf_path: str = None) -> dict:
    '''
    rdf: rdflib.Graph object for analysis
    rdf_path: path to .rdf graph

    returns a dictionary of metadata for RDF
    '''

    # get max width of tree
    def max_width(subs):
        count_dict = Counter(subs)
        max = 0
        for k in count_dict.keys():
            if count_dict[k] > max:
                max = count_dict[k]

        return max

    # return depth of tree
    def depth(g, leaves):
    
        depth = 0
        # remove leaves from graph, find new leaves and increment depth by 1
        while len(list(g.objects())) > 0:

            for leaf in leaves:
                g.remove((None, None, leaf))
            leaves = list(set(g.objects()) - set(g.subjects()))
            depth = depth + 1
            if depth > 100:
                return "Error"
        # print("Depth ", depth)
        return depth

    g = None
    meta = dict()

    if rdf is not None:
        g = rdf
    elif rdf_path is not None:
        meta['name'] = get_filename(rdf_path)
        meta['path'] = rdf_path
        try:
            g = get_rdfGraph(rdf_path)
        except:
            print(f"There was an error parsing {rdf_path}. Is this a valid filepath?")

    assert g is not None, "You must pass a valid rdflib.Graph object or path to an RDF file"

    # Graph Stats
    meta['num_relations'] = len(g)                       # Num edges (and relations) in Graph
    meta['objects'] = list(g.objects())                  # List of Objects
    meta['subjects'] = list(g.subjects())                # List of Subjects
    meta['nodes'] = set (objects + subjects)             # Set of Nodes (Entities)
    meta['leaves'] = list(set(objects)-set(subjects))    # List of leaves (degree 1) - the set of objects that are not also subjects
    meta['root'] = list(set(subjects) - set(objects))    # Should only be 1, but made list just in case. Set of subjects that are not also objects
    meta['num_nodes'] = len(nodes)                       # Number of Nodes(Entities) in Graph
    meta['num_leaves'] = len(leaves)                     # Number of leaves
    meta['max_width'] = max_width(subjects)              # Max width of tree (most children in single node)
    meta['connected'] = g.connected()                    # Is graph connected?
    meta['depth'] = depth(g, meta['leaves'])             # Depth of Graph

    return meta