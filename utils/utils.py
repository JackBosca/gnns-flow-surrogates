def decompose_graph(graph):
    '''
    Decomposes a torch_geometric.data.Data object into its components.
    '''
    # initialize components
    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys():
        if key == "x":
            x = graph.x
        elif key == "edge_index":
            edge_index = graph.edge_index
        elif key == "edge_attr":
            edge_attr = graph.edge_attr
        elif key == "global_attr":
            global_attr = graph.global_attr
        else:
            pass
    return (x, edge_index, edge_attr, global_attr)