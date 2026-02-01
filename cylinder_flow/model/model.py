"""
Encode-process-decode GNN for 2D cylinder-flow velocity updates.

* Encoder: MLPs map raw node/edge features -> latent H.
* Processor: K x (EdgeBlock -> NodeBlock) message-passing layers with
  residuals. Each layer keeps node/edge size = H.
* Decoder: MLP maps node latent H -> output (2D velocity delta).
* Simulator: wraps model, handles normalization, training vs inference,
  noise injection, save/load.

## Data layout (PyG `Data`)

* graph.x        : [N, Din]  raw node features (e.g. [type_onehot, vx, vy])
* graph.edge_attr: [E, Dein] raw edge features
* graph.edge_index: [2, E]   senders = edge_index[0], receivers = edge_index[1]
* Encoder -> outputs x: [N, H], edge_attr: [E, H]
* Decoder -> outputs: [N, 2]

## Per-block operations (GNNBlock)

1. Edge update (EdgeBlock): for each edge e (i->j)
   * collected = [x[i], x[j], edge_attr[e]]  # -> [3H]
   * edge_attr'[e] = edge_mlp(collected)     # -> [H]

2. Node update (NodeBlock): for each node j
   * agg = sum(edge_attr' for edges with receiver j)  # -> [H]
   * collected = [x[j], agg]                          # -> [2H]
   * x'[j] = node_mlp(collected)                      # -> [H]

* N: nodes, E: edges, H: hidden_size
* Raw: x [N, Din], edge_attr [E, Dein]
* Encoded: x [N, H], edge_attr [E, H]
* Output: [N, 2]

## Training vs Inference (Simulator.forward)

* Training: add `velocity_sequence_noise` to frames, build node_attr,
  predict normalized delta, return (pred, normalized_target_delta).
* Inference: build node_attr from frames, predict, denormalize delta,
  return frames + delta (predicted velocity).
"""

import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
from .blocks import EdgeBlock, NodeBlock, build_mlp
from utils.utils import decompose_graph, copy_geometric_data, Normalizer

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

class Simulator(nn.Module):
    '''
    The full simulator model with normalization, noise, and training/inference modes.
    '''
    def __init__(self, message_passing_num, node_input_size, edge_input_size, device, model_dir='checkpoint/simulator.pth'):
        super(Simulator, self).__init__()

        self.node_input_size =  node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, node_input_size=node_input_size, edge_input_size=edge_input_size).to(device)
        self._output_normalizer = Normalizer(size=2, name='output_normalizer', device=device)
        self._node_normalizer = Normalizer(size=node_input_size, name='node_normalizer', device=device)
        # self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        print('Simulator model initialized.')

    def update_node_attr(self, frames, types:torch.Tensor):
        """Update node attributes with normalized velocity and one-hot encoded node types."""
        node_feature = []

        # append velocity and one-hot encoded node types to node features
        node_feature.append(frames)
        node_type = torch.squeeze(types.long())
        # NORMAL: 0, OBSTACLE = 1, AIRFOIL = 2, HANDLE = 3, INFLOW = 4, OUTFLOW = 5, WALL_BOUNDARY = 6, SIZE = 9
        one_hot = torch.nn.functional.one_hot(node_type, 9)
        node_feature.append(one_hot)

        # concatenate all node features along feature dimension and normalize
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, self.training)

        return attr

    def velocity_delta(self, noised_frames, next_velocity):
        """Compute the velocity delta between the next velocity and the noised current frames."""
        delta = next_velocity - noised_frames
        return delta

    def forward(self, graph:Data, velocity_sequence_noise):
        # extract node_type and velocities from graph.x
        node_type = graph.x[:, 0:1]
        frames = graph.x[:, 1:3]

        if self.training:
            target = graph.y

            noised_frames = frames + velocity_sequence_noise
            node_attr = self.update_node_attr(noised_frames, node_type)
            graph.x = node_attr

            # model prediction
            predicted = self.model(graph)

            target_delta = self.velocity_delta(noised_frames, target)
            target_delta_normalized = self._output_normalizer(target_delta, self.training)

            return predicted, target_delta_normalized

        else:
            node_attr = self.update_node_attr(frames, node_type)
            graph.x = node_attr

            # model prediction
            predicted = self.model(graph)

            velocity_update = self._output_normalizer.inverse(predicted)
            predicted_velocity = frames + velocity_update

            return predicted_velocity

    def save_checkpoint(self, save_dir=None):
        if save_dir is None:
            save_dir = self.model_dir

        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        # save model state_dict and normalizer parameters
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer  = self._node_normalizer.get_variable()

        to_save = {'model':model, '_output_normalizer':_output_normalizer, '_node_normalizer':_node_normalizer}

        torch.save(to_save, save_dir)
        print('Simulator model saved at %s.' % save_dir)

    def load_checkpoint(self, ckp_dir=None):
        if ckp_dir is None:
            ckp_dir = self.model_dir
        
        dicts = torch.load(ckp_dir)
        self.load_state_dict(dicts['model'])

        # remove 'model' key to avoid redundant loading
        keys = list(dicts.keys())
        keys.remove('model')

        # load normalizer parameters
        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.' + k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s." % ckp_dir)
