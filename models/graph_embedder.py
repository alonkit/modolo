import torch
from torch_geometric.data import HeteroData
from models.layers import AtomEncoder, GaussianSmearing
import torch.nn as nn


class GraphEmbedder(nn.Module):
    @staticmethod
    def _to_key(*args):
        return "_".join(args)
    def __init__(
        self,
        distance_embed_dim,
        cross_distance_embed_dim,
        lig_max_radius,
        rec_max_radius,
        cross_max_distance,
        lig_feature_dims,
        lig_edge_feature_dim,
        lig_emb_dim,
        rec_feature_dims,
        atom_feature_dims,
        prot_emd_dim,
        dropout,
        lm_embedding_dim,
    ):
        super().__init__()

        self.distance_expansion = nn.ModuleDict(
            {
                "ligand": GaussianSmearing(0.0, lig_max_radius, distance_embed_dim),
                "atom": GaussianSmearing(0.0, lig_max_radius, distance_embed_dim),
                "receptor": GaussianSmearing(0.0, rec_max_radius, distance_embed_dim),
                self._to_key("atom", "receptor"): GaussianSmearing(
                    0.0, rec_max_radius, distance_embed_dim
                ),
                "cross": GaussianSmearing(
                    0.0, cross_max_distance, cross_distance_embed_dim
                ),
            }
        )

        atom_encoder_class = AtomEncoder
        self.node_embedders = nn.ModuleDict(
            {
                "ligand": atom_encoder_class(
                    emb_dim=lig_emb_dim, feature_dims=lig_feature_dims
                ),
                "receptor": atom_encoder_class(
                    emb_dim=prot_emd_dim,
                    feature_dims=rec_feature_dims,
                    lm_embedding_dim=lm_embedding_dim,
                ),
                "atom": atom_encoder_class(
                    emb_dim=prot_emd_dim, feature_dims=atom_feature_dims
                ),
            }
        )
        self.edge_embedders = nn.ModuleDict(
            {
                self._to_key("ligand", "ligand"): self.create_edge_embedding(
                    in_dim=lig_edge_feature_dim + distance_embed_dim,
                    out_dim=lig_emb_dim,
                    dropout=dropout,
                ),
                self._to_key("ligand", "atom"): self.create_edge_embedding(
                    in_dim=cross_distance_embed_dim,
                    out_dim=lig_emb_dim,
                    dropout=dropout,
                ),
                self._to_key("ligand", "receptor"): self.create_edge_embedding(
                    in_dim=cross_distance_embed_dim,
                    out_dim=lig_emb_dim,
                    dropout=dropout,
                ),
                self._to_key("receptor", "receptor"): self.create_edge_embedding(
                    in_dim=distance_embed_dim, out_dim=prot_emd_dim, dropout=dropout
                ),
                self._to_key("atom", "atom"): self.create_edge_embedding(
                    in_dim=distance_embed_dim, out_dim=prot_emd_dim, dropout=dropout
                ),
                self._to_key("atom", "receptor"): self.create_edge_embedding(
                    in_dim=distance_embed_dim, out_dim=prot_emd_dim, dropout=dropout
                ),
            }
        )

    def create_edge_embedding(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, data: HeteroData):
        nodes, edges = data.metadata()
        for node in nodes:
            if node in self.node_embedders:
                data[node].x = self.node_embedders[node](data[node].x)

        for src, _, dst in edges:
            edge_type = (src, dst)
            if src == dst:
                dist_expander = self.distance_expansion[src]
            elif edge_type in self.distance_expansion:
                dist_expander = self.distance_expansion[self._to_key(*edge_type)]
            else:
                dist_expander = self.distance_expansion["cross"]
            src_idx, dst_idx = data[edge_type].edge_index
            src_pos, dst_pos = data[src].pos[src_idx], data[dst].pos[dst_idx]
            dist_exp = dist_expander(
                (src_pos - dst_pos).norm(dim=-1)
            )  # euclidean distance
            if (edge_attr := data[edge_type].get("edge_attr")) is not None:
                dist_exp = torch.cat([dist_exp, edge_attr], dim=-1)
            data[edge_type].edge_attr = self.edge_embedders[self._to_key(*edge_type)](dist_exp)

        return data
