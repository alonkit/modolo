import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor
from torch import nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import EdgeType, Metadata, NodeType
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


class PGHTConv2(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        edge_in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        group: str = "sum",
        num_attn_groups: int = 2,
        dropout = 0.1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        assert out_channels % num_attn_groups == 0 , 'out_channels must be divisible by num_attn_groups'

        self.in_channels = in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels
        self.group = group
        self.num_attn_groups = num_attn_groups
        full_edge_in_ch = in_channels*2 + edge_in_channels
        self.k_lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, affine=False),
            nn.ReLU(),
        )
        self.q_lin = nn.Sequential(
            nn.Linear(full_edge_in_ch, out_channels),
            nn.BatchNorm1d(out_channels, affine=False),
            nn.ReLU(),
        )
        self.v_lin = Linear(full_edge_in_ch, out_channels)

        self.weight_encoding = nn.Sequential(
            nn.Linear(out_channels, num_attn_groups),
            nn.BatchNorm1d(num_attn_groups),
            nn.ReLU(),
            nn.Linear(num_attn_groups, num_attn_groups)
        )

        self.out_lin = Linear(out_channels, out_channels)
        self.identity_lin = Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.norm_input = nn.BatchNorm1d(in_channels)
        self.norm_output = nn.BatchNorm1d(out_channels)

        self.reset_parameters()
        self.attn_drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    # nn.Sequential(nn.Linear(cross_distance_embed_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout),nn.Linear(emb_dim, emb_dim))
    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.out_lin)

    def forward(
        self,
        node_attr: Tensor,
        edge_index: Union[Tensor,SparseTensor],
        edge_attr: Tensor,
        coords: Tensor,
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
        node_attr = self.act(node_attr)
        
        # propagate_type: (v: Tensor, coords: Tensor, e: Tensor)
        out = self.propagate(
                edge_index,
                size=None,
                v=node_attr,
                coords=coords,
                e=edge_attr,
            )

        out = self.out_lin(F.relu(out))
        out = self.norm_output(out)
        identity = node_attr
        if out.size(-1) != identity.size(-1):
            identity = self.identity_lin(identity)

        out = out + identity
        out = self.act(out)
        return out

    def message(self, 
                v_i: Tensor,v_j: Tensor,
                e: Tensor, coords_i: Tensor,coords_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        v_i_e_v_j = torch.concat([v_i, e, v_j],dim=-1) # maybe through coords here?
        # dists = (coords_i - coords_j).norm()
        k = self.k_lin(v_i)
        q = self.q_lin(v_i_e_v_j)
        v = self.v_lin(v_i_e_v_j)

        weight = self.attn_drop(self.weight_encoding(q - k))

        weight = softmax(weight, index, ptr, size_i)
        out = (v).view(-1, self.num_attn_groups, self.out_channels // self.num_attn_groups) * weight.unsqueeze(-1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels})')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels})')
