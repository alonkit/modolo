from typing import List, Tuple, Union
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


from models.layers.point_graph_transformer_conv import PGHTConv
from models.layers.point_graph_transformer_conv_v2 import PGHTConv2


class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        hidden_channels: Union[int, List[int]],
        out_channels: int,
        attention_groups: Union[int, List[int]],
        graph_embedder: torch.nn.Module,
        dropout: float=0.1,
        num_layers: int = None,
        max_length=128,
        version:int = 1
    ):

        assert (not isinstance(hidden_channels, int)) or num_layers, "Either hidden_channels is a list or num_layers must be provided"
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * (num_layers - 1)
        super(GraphEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        num_channels = [in_channels,*hidden_channels,out_channels]
        if isinstance(attention_groups, int):
            attention_groups = [attention_groups] * (len(num_channels) - 1)
        
        
        for i, (in_channels, out_channels, attn_groups) in enumerate(zip(num_channels[:-1], num_channels[1:], attention_groups)):
            self.convs.append(
                
                PGHTConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    edge_in_channels=edge_channels,
                    num_attn_groups=attn_groups,
                    dropout=dropout,
                ) if version == 1 else
                PGHTConv2(
                    in_channels=in_channels,
                    edge_in_channels=edge_channels,
                    out_channels=out_channels,
                    num_attn_groups=attn_groups,
                    dropout=dropout
                )
            )
        self.edge_channels = edge_channels
        self.graph_embedder = graph_embedder
        self.max_length = max_length
        self.in_channels = out_channels
        self.out_channels = out_channels
        
        self.freeze_layers = [graph_embedder, *self.convs ]

    def get_new_indexes_after_masking(self,graph, idxs, molecule_sidechain_mask_idx):
        full_idxs = torch.arange(graph['ligand'].x.shape[0], device=idxs.device)
        mask = (graph.sidechains_mask < molecule_sidechain_mask_idx).to(full_idxs.device)
        full_idxs_after_masking = full_idxs[mask]
        return torch.searchsorted(full_idxs_after_masking,idxs)
        

    
    def mask_graph_sidechains(self, graph, molecule_sidechain_mask_idx):
        device = graph['ligand'].x.device
        masks = {
            node_t:(
                (graph.sidechains_mask < molecule_sidechain_mask_idx).to(device)
                if node_t == "ligand"
                else torch.arange(graph[node_t].num_nodes, device=device)
            )
            for node_t in graph.metadata()[0]
        }
        graph=  graph.subgraph(masks)
        batch = graph['ligand'].batch
        ptr = torch.arange(batch.shape[0]-1, device=batch.device) + 1
        change = batch[:-1] != batch[1:]
        ptr = torch.tensor([0, *ptr[change], batch.shape[0]], device=batch.device)
        graph['ligand'].ptr = ptr
        return graph

    def pred_distances(self, data):
        data = self.forward(data, keep_hetrograph=True)
        ll_i, ll_j = data['ligand'].x[data['ligand'].edge_index]
        
        v_i, v_j = data.x[data.edge_index]
        v_i_e_v_j = torch.concat([v_i, data.edge_index, v_j],dim=-1)
        pred_dists = self.dist_final_layer(v_i_e_v_j)
        return pred_dists

    def dist_forward(self, hdata: HeteroData):
        hdata = self.forward(hdata, keep_homograph=True)
        noise_pred = self.dist_final_layer(hdata['ligand'].x)
        return noise_pred

    def forward(self, hdata: HeteroData, keep_hetrograph=False,keep_homograph=False,convs=None):
        hdata = self.graph_embedder(hdata)
        hdata = ToUndirected()(hdata)
        data = hdata.to_homogeneous()
        x = data.x
        for conv in (convs or self.convs):
            x = conv(x, data.edge_index, data.edge_attr, data.pos)
        data.x = x
        if keep_homograph:
            return data
        data = data.to_heterogeneous()
        if keep_hetrograph:
            return data
        output = data['ligand'].x
        batch_indices = data['ligand'].batch
        batch_size = batch_indices.max().item() + 1
        emb_dim = output.size(1)

        res = torch.zeros(batch_size,self.max_length,emb_dim).to(output.device)
        res[self._graph_batch_indices_to_sequence(batch_indices)] = output
        return res

    def _graph_batch_indices_to_sequence(self, batch_indices: torch.Tensor):
        change_indices = torch.nonzero(batch_indices[1:] != batch_indices[:-1]).flatten() + 1
        dist_between_change = change_indices.clone()
        dist_between_change[1:] = change_indices[:-1] - change_indices[1:]
        dist_between_change[0] = -dist_between_change[0]
        dist_between_change = dist_between_change + 1
        jumps = torch.ones_like(batch_indices)
        jumps[change_indices] = dist_between_change
        batch_range = torch.cumsum(jumps,0) - 1
        return batch_indices,batch_range

    def create_memory_key_padding_mask(self,data:HeteroData):
        batch_indices = data['ligand'].batch
        batch_size = batch_indices.max().item() + 1
        mask = torch.zeros(batch_size,self.max_length).to(batch_indices.device) + 1
        mask[self._graph_batch_indices_to_sequence(batch_indices)] = 0
        return mask.bool()
