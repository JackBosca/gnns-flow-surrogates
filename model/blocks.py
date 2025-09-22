import torch
import torch.nn as nn
from torch_scatter import scatter_add
from utils.utils import decompose_graph
from torch_geometric.data import Data

class EdgeBlock(nn.Module):
    '''
    Edge update block.
    '''
    def __init__(self, custom_func=None):
        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        # retrieve graph components
        node_attr, edge_index, edge_attr, _ = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)

        # pass through the network to update edge attributes
        edge_attr_ = self.net(collected_edges)

        # return updated graph
        return Data(x=node_attr, edge_attr=edge_attr_, edge_index=edge_index)
    
class NodeBlock(nn.Module):
    '''
    Node update block.
    '''
    def __init__(self, custom_func=None):
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        # retrieve graph edge attributes and number of nodes
        edge_attr = graph.edge_attr
        num_nodes = graph.num_nodes
        _, receivers_idx = graph.edge_index

        # aggregate edge attributes for each node
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect = []
        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        # pass through the network to update node attributes
        x = self.net(collected_nodes)

        # return updated graph
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)
       