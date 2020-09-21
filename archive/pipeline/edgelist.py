from rdf_to_graph import rdf2Graph

filename = snakemake.input[0]
try:
    graph = rdf2Graph(filename)

    print(graph.uid)

    graph.getEdgelist(snakemake.output[0])
    graph.getEdgeMap(snakemake.output[1])

except:
    print("something failed")


# if __name__ == "__main__":
#     filename = snakemake.input[0]

#     graph = rdf2Graph(filename)

#     print(graph.uid)

#     graph.getEdgelist(snakemake.output[0])
#     graph.getEdgeMap(snakemake.output[1])

#     print("Node 3: ", graph.node_content(3) )
#     print("Node 10: ", graph.node_content(10) )
#     print(graph.edge_content(4))
#     print(graph.edge_content(9))
#     test = graph.edge_content(9)
#     print(graph.edge_value(test))
    

#     for NI in graph.snap['graph'].Nodes():
#         print("node: %d, out-degree %d, in-degree %d" % ( NI.GetId(), NI.GetOutDeg(), NI.GetInDeg()))