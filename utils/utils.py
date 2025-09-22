from torch_geometric.data import Data

# def decompose_graph(graph):
#     '''
#     Decomposes a torch_geometric.data.Data object into its components.
#     '''
#     # initialize components
#     x, edge_index, edge_attr, global_attr = None, None, None, None
#     for key in graph.keys():
#         if key == "x":
#             x = graph.x
#         elif key == "edge_index":
#             edge_index = graph.edge_index
#         elif key == "edge_attr":
#             edge_attr = graph.edge_attr
#         elif key == "global_attr":
#             global_attr = graph.global_attr
#         else:
#             pass
#     return (x, edge_index, edge_attr, global_attr)

def decompose_graph(graph):
    '''
    Decompose a PyG Data object into main attributes, cloning them.
    '''
    x = graph.x.clone() if hasattr(graph, 'x') and graph.x is not None else None
    edge_index = graph.edge_index.clone() if hasattr(graph, 'edge_index') and graph.edge_index is not None else None
    edge_attr = graph.edge_attr.clone() if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
    global_attr = graph.global_attr.clone() if hasattr(graph, 'global_attr') and graph.global_attr is not None else None
    
    return x, edge_index, edge_attr, global_attr


def copy_geometric_data(graph):
    '''
    Return a copy of torch_geometric.data.data.Data object.
    '''
    node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)

    g_copy = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    g_copy.global_attr = global_attr

    return g_copy
