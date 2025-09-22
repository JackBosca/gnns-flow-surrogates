import torch.nn as nn
from torch_geometric.data import Data
from .blocks import EdgeBlock, NodeBlock, build_mlp
from utils.utils import decompose_graph, copy_geometric_data

class Encoder(nn.Module):
    '''
    Encode the node and edge features to latent space.
    '''
    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128):
        super(Encoder, self).__init__()

        # build edge and node encoder mlps
        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):
        node_attr, _, edge_attr, _ = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)
        
        # return encoded graph
        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)

class GNNBlock(nn.Module):
    '''
    A single GNN block with one EdgeBlock and one NodeBlock.
    '''
    def __init__(self, hidden_size=128):
        super(GNNBlock, self).__init__()

        # define input dimension for edges: 3*hidden_size (sender features, receiver features, edge_attr)
        eb_input_dim = 3 * hidden_size

        # define input dimension for nodes: 2*hidden_size (node features, aggregated edge features)
        nb_input_dim = 2 * hidden_size

        # build edge and node MLPs
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
        # copy current graph for residual connection
        graph_last = copy_geometric_data(graph)

        # pass graph through EdgeBlock (update graph.edge_attr) and NodeBlock (update graph.x)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)

        # residual connection for both edge features and node features
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x

        # return updated graph after one GNN layer with residual connection
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)

class Decoder(nn.Module):
    '''
    Decode the node features to the target space.
    '''
    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)

class EncoderProcesserDecoder(nn.Module):
    '''
    The full model with an encoder, multiple GNN blocks, and a decoder.
    '''
    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):
        super(EncoderProcesserDecoder, self).__init__()

        # build encoder
        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size)
        
        # build processer with multiple GNN blocks
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GNNBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)
        
        # build decoder
        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph):
        # pass graph through encoder, processer (multiple GNN blocks), and decoder
        graph= self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded
    