import snap
from snap import TNEANet
from snap import TStrIntH
import os
import rdflib
from collections import Counter
import ntpath

def build_snap_graph(filename):
    graph = TNEANet.New()

    rdf = rdflib.Graph().parse(filename)

    node_values = list(set( list(rdf.objects()) + list(rdf.subjects()) ))

    # hash table for storing rdf info
    nodes = TStrIntH()
    edges = {}

    # add all nodes
    for idx, node in enumerate(node_values):
        nodes[node] = idx
        graph.AddNode(idx)

    # add all edges
    count = 0
    for s, p, o in rdf:
        edges[count] = p
        graph.AddEdge(nodes[s], nodes[o], count)
        count += 1
    
    return graph, nodes, edges


class rdf2Graph:

    '''
    ATTRIBUTES
    --------------------------
    uid : Unique ID
    rdf = { RDF info
            graph
            objects
            subjects
            nodes
            leaves
            root
        }
    num_edges
    num_nodes
    num_leaves
    connected
    max_width
    depth
    snap = { SNAP info
        nodes
        edges
        graph
    }

    '''

    def __init__(self, rdf_path):
        assert os.path.isfile(rdf_path)

        ### Pull all metadata from RDFLIB generated graph ###
        
        graph = rdflib.Graph().parse(rdf_path)
        objects = list(graph.objects())
        subjects = list(graph.subjects())
        num_edges = len(graph)
        nodes = list (set (objects + subjects))
        leaves = list( set(objects) - set(subjects))
        root = list(set(subjects) - set(objects))
        num_nodes = len(nodes)
        num_leaves = len(leaves)
        connected = graph.connected()
        max_width = 0
        depth = 0

        # determine maximum width
        count_dict = Counter(subjects)
        for k in count_dict.keys():
            if count_dict[k] > max_width:
                max_width = count_dict[k]

        # determine the depth by removing all leaves, and counting 1 if not at root
        temp = rdflib.Graph().parse(rdf_path)
        temp_leaves = list( set(temp.objects()) - set(temp.subjects()))
        while len(list(temp.objects())) > 0:
            for leaf in temp_leaves:
                temp.remove((None, None, leaf))
            temp_leaves = list(set(temp.objects()) - set(temp.subjects()))
            depth += 1
            if depth > 100:
                depth = -1

        
        self.rdf = {
            'graph' : graph,
            'objects' : objects,
            'subjects' : subjects,
            'nodes' : nodes,
            'leaves': leaves,
            'root': root
        }
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.num_leaves = num_leaves
        self.connected = connected
        self.max_width = max_width
        self.depth = depth
        self.uid = ntpath.basename(rdf_path).replace('.rdf', '')

        ### Build SNAP Graph for node2vec/SNAP operations ###
        snap_graph, snap_nodes, snap_edges = build_snap_graph(rdf_path)
        self.snap = {
            'nodes': snap_nodes,
            'edges' : snap_edges,
            'graph': snap_graph,
            'edgelist' : None
        }

    def node_content(self, number):
        return self.rdf['nodes'][number]

    def node_value(self, content):
        return self.snap['nodes'][content]

    def edge_content(self, number):
        return self.snap['edges'][number]

    def edge_value(self, content):
        keys = []
        for key, value in self.snap['edges'].items():
            if value == content:
                keys.append(key)
        return keys

    def getEdgelist(self, filepath):
        snap.SaveEdgeList(self.snap['graph'], filepath)

if __name__ == "__main__":
    filename = "../examples/test.rdf"

    graph = rdf2Graph(filename)

    print(graph.uid)

    graph.getEdgelist('../examples/test.edgelist')

    # print("Node 3: ", graph.node_content(3) )
    # print("Node 10: ", graph.node_content(10) )
    # print(graph.edge_content(4))
    # print(graph.edge_content(9))
    # test = graph.edge_content(9)
    # print(graph.edge_value(test))
    

    # for NI in graph.snap['graph'].Nodes():
    #     print("node: %d, out-degree %d, in-degree %d" % ( NI.GetId(), NI.GetOutDeg(), NI.GetInDeg()))