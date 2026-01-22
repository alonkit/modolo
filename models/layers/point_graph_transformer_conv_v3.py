import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.utils import softmax, scatter


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class PGHTConv3(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        edge_in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        group: str = "sum",
        num_attn_groups: int = 2,
        dropout = 0.1,
        distance_scaling: bool = False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        assert out_channels % num_attn_groups == 0 , 'out_channels must be divisible by num_attn_groups'
        self.dropout = nn.Dropout(dropout)
        self.in_channels = in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels
        self.group = group
        self.num_attn_groups = num_attn_groups
        full_edge_in_ch = in_channels*2 + edge_in_channels
        self.k_lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, affine=False),
            self.dropout,
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            
        )
        self.q_lin = nn.Sequential(
            nn.Linear(full_edge_in_ch, out_channels),
            nn.BatchNorm1d(out_channels, affine=False),
            self.dropout,
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.v_lin = Linear(full_edge_in_ch, out_channels)

        self.weight_encoding = nn.Sequential(
            nn.Linear(out_channels, num_attn_groups),
            nn.BatchNorm1d(num_attn_groups),
            self.dropout,
            nn.SiLU(),
            nn.Linear(num_attn_groups, num_attn_groups)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            self.dropout,
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

            

        self.identity_lin = Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.norm_input = nn.BatchNorm1d(in_channels)

        self.reset_parameters()
        self.act = nn.ReLU()
        
        self.distance_scaling = distance_scaling

    # nn.Sequential(nn.Linear(cross_distance_embed_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout),nn.Linear(emb_dim, emb_dim))
    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.node_mlp)

    def forward(
        self,
        node_attr: Tensor,
        edge_index: Union[Tensor,SparseTensor],
        edge_attr: Tensor,
        coords: Tensor,
        data: Data
    ) -> Tensor:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The output node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """

        node_attr = self.norm_input(node_attr)
        # node_attr = self.act(node_attr)
        
        # propagate_type: (v: Tensor, coords: Tensor, e: Tensor)
        out = self.propagate(
                edge_index,
                size=None,
                v=node_attr,
                coords=coords,
                e=edge_attr,
            )
        
        
        
        out = self.node_mlp(out)
        identity = node_attr
        if out.size(-1) != identity.size(-1):
            identity = self.identity_lin(identity)

        out = out + identity
        # out = self.act(out)
        return out


    def message(self, 
                v_i: Tensor,v_j: Tensor,
                e: Tensor, coords_i: Tensor,coords_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        v_i_e_v_j = torch.concat([v_i, e, v_j],dim=-1) # maybe through coords here?
        
        k = self.k_lin(v_i)
        q = self.q_lin(v_i_e_v_j)
        v = self.v_lin(v_i_e_v_j)

        weight = self.weight_encoding(q - k)
        if self.distance_scaling:
            dists = (coords_i - coords_j).norm(dim=-1, keepdim=True)
            weight  *= 1/(1+dists)
        weight = softmax(weight, index, ptr, size_i)
            
        
        out = (v).view(-1, self.num_attn_groups, self.out_channels // self.num_attn_groups) * weight.unsqueeze(-1)
        out = out.view(-1, self.out_channels)
       
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels})')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels})')
