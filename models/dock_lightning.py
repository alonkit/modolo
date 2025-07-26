from collections import defaultdict
import copy
from datetime import datetime
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.custom_distributed_sampler import CustomDistributedSampler
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from datasets.process_chem.process_sidechains import (
    calc_tani_sim,
    get_fp,
    reconstruct_from_core_and_chains,
)
from models.modolo import Modolo
from models.graph_encoder import GraphEncoder
from utils.logging_utils import configure_logger, get_logger


class DockLightning(pl.LightningModule):
    def __init__(
        self,
        graph_encoder_model: GraphEncoder,
        lr,
        weight_decay,
        name=None,
        smol=True,
        max_noise_scale=5.
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.graph_encoder_model = graph_encoder_model
        edge_c = graph_encoder_model.edge_channels
        g_out = graph_encoder_model.out_channels
        self.distances_layer = nn.Sequential(nn.Linear(g_out*2+edge_c, g_out*2+edge_c),nn.Dropout(0.1), nn.ReLU(), nn.Linear(g_out*2+edge_c, 1))
        self.name = name or f'{datetime.today().strftime("%Y-%m-%d-%H_%M_%S")}'
        self.name = f'dock_{self.name}'
        self.smol = smol
        self.max_noise_scale = max_noise_scale

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.sub_proteins.open()

    
    def t_to_sigma(self, t):
        return 0.05 ** (1-min(t,1)) * self.max_noise_scale ** t
    
    def pred_distances(self, data):
        data = self.graph_encoder_model(data, keep_hetrograph=True)
        device = data['ligand'].x.device
        masks = {
            node_t:(
                (data.sidechains_mask != 0).to(device)
                if node_t == "ligand"
                else torch.arange(data[node_t].num_nodes, device=device)
            )
            for node_t in data.metadata()[0]
        }
        data=  data.subgraph(masks)
        ll = data['ligand','ligand'].edge_index
        ligand_edges_no_self = ll[:, ll[0]!=ll[1]]
        ll_i, ll_j = data['ligand'].x[ligand_edges_no_self]
        lr_i, lr_j = data['ligand'].x[data['ligand', 'receptor'].edge_index[0]], data['receptor'].x[data['ligand', 'receptor'].edge_index[1]]
        la_i, la_j = data['ligand'].x[data['ligand', 'atom'].edge_index[0]], data['atom'].x[data['ligand', 'atom'].edge_index[1]]
        ll = torch.concat([ll_i, ll_j, data['ligand','ligand'].edge_attr],dim=-1)
        lr = torch.concat([lr_i, lr_j, data['ligand','receptor'].edge_attr],dim=-1)
        la = torch.concat([la_i, la_j, data['ligand','atom'].edge_attr],dim=-1)
        edges = torch.concat([ll,lr,la], dim=0)
        pred_dists = self.distances_layer(edges).squeeze(-1)
        orig_dists = self.get_distances(data, data['ligand'].orig_pos)
        return pred_dists, orig_dists

    def get_distances(self,data, lig_poses):
        ll_i, ll_j = lig_poses[data['ligand','ligand'].edge_index]
        lr_i, lr_j = lig_poses[data['ligand', 'receptor'].edge_index[0]], data['receptor'].pos[data['ligand', 'receptor'].edge_index[1]]
        la_i, la_j = lig_poses[data['ligand', 'atom'].edge_index[0]], data['atom'].pos[data['ligand', 'atom'].edge_index[1]]
        ll = (ll_i - ll_j).norm(dim=-1)
        lr = (lr_i - lr_j).norm(dim=-1)
        la = (la_i - la_j).norm(dim=-1)
        return torch.concat([ll,lr,la], dim=0)

    def hide_sidechains(self, data):
        orig_pos = data['ligand'].pos.clone()
        data['ligand'].orig_pos = orig_pos
        mask = data.sidechains_mask != 0
        sigma = self.t_to_sigma(self.current_epoch / self.trainer.max_epochs)
        pos_noise = torch.normal(0,sigma, orig_pos[mask].shape,device=orig_pos.device)
        data['ligand'].pos[mask] = orig_pos[mask] + pos_noise
        # for i in range(len(data)):
        #     mol = data[i]
        #     for i, hole_neighbor in enumerate(mol.hole_neighbors):
        #         hole_pos = mol['ligand'].pos[hole_neighbor].clone()
        #         mol['ligand'].pos[mol.sidechains_mask == i+1] = hole_pos
        return data

    def get_loss(self, data):
        self.hide_sidechains(data)
        pred_dists, orig_dists = self.pred_distances(data)
        loss = ((orig_dists - pred_dists)**2 ) * (1/(orig_dists+1))
        loss = loss.mean()
        return loss
         
    
    def training_step(self, data, batch_idx):
        loss = self.get_loss(data)
        self.log("train_noise_loss", loss, sync_dist=True, prog_bar=True)
        return loss
            
            
    def validation_step(self, data, batch_idx):
        loss = self.get_loss(data)
        self.log("val_noise_loss", loss, batch_size=len(data), sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
