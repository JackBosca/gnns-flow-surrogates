import torch
import torch.nn as nn
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

class Normalizer(nn.Module):
    '''
    Normalizes input data and accumulates statistics (mean, std) during training.
    '''
    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8, name='Normalizer', device='cuda'):
        super(Normalizer, self).__init__()

        self.name=name
        # max number of batch to accumulate statistics for, to prevent overflow
        self._max_accumulations = max_accumulations
        # constant to prevent division by zero
        self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=device)
        # total number of samples accumulated so far
        self._acc_count = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        # how many batches have been accumulated so far
        self._num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        # sum and square sum for mean and std computation
        self._acc_sum = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
        self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)

    def forward(self, batch_data, accumulate=True):
        """Normalize input data and accumulate statistics."""
        if accumulate:
        # stop accumulating after a million updates, to prevent accuracy issues
            if self._num_accumulations < self._max_accumulations:
                # detach data so that accumulating does not influence gradients
                self._accumulate(batch_data.detach())
        return (batch_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batch_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = batch_data.shape[0]
        data_sum = torch.sum(batch_data, axis=0, keepdims=True)
        squared_data_sum = torch.sum(batch_data**2, axis=0, keepdims=True)

        # update accumulators counts
        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean()**2)
        return torch.maximum(std, self._std_epsilon)

    def get_variable(self):
        dict = {'_max_accumulations':self._max_accumulations,
        '_std_epsilon':self._std_epsilon,
        '_acc_count': self._acc_count,
        '_num_accumulations':self._num_accumulations,
        '_acc_sum': self._acc_sum,
        '_acc_sum_squared':self._acc_sum_squared,
        'name':self.name
        }

        return dict
    