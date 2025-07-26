import torch
from rdkit import Chem
import torch.nn as nn
import torch.nn.functional as F


class Modolo(nn.Module):
    def __init__(
        self,
        transformer_encoder,
        transformer_decoder,
        interaction_encoder,
        graph_encoder,
        add_hole_heighbours=False,
        use_receptors=True
    ):
        super(Modolo, self).__init__()
        self.add_hole_heighbours = add_hole_heighbours
        self.text_encoder = transformer_encoder
        self.graph_encoder = graph_encoder
        self.decoder = transformer_decoder
        self.interaction_encoder = interaction_encoder
        self.use_receptors = use_receptors
        # self.linear = nn.Linear(num_gnn_features, d_model)
        if self.graph_encoder is not None:
            self.freeze_layers = [*self.graph_encoder.freeze_layers, [self.text_encoder, self.interaction_encoder]]
        else:
            self.freeze_layers = []

    def _train_decode(self, tgt, memory, memory_key_padding_mask):
        target_mask, target_padding_mask = self.decoder.create_target_masks(tgt)
        output = self.decoder(
            tgt,
            memory,
            target_mask,
            target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output

    def _create_text_memory(self, smiles_tokens_src):
        if self.text_encoder is None:
            return None, None
        smiles_padding_mask = self.text_encoder.create_src_key_padding_mask(
            smiles_tokens_src
        )
        smiles_memory = self.text_encoder(smiles_tokens_src, smiles_padding_mask)
        if (
            smiles_memory.shape[1] == smiles_padding_mask.shape[-1]
        ):  # transformer encoder did not take the fast path
            smiles_memory = smiles_memory[:, ~smiles_padding_mask.all(0)]

        smiles_padding_mask = smiles_padding_mask[:, ~smiles_padding_mask.all(0)]
        return smiles_memory, smiles_padding_mask

    def _get_local_clusters_vecs(self, g, cluster_centers_idxs, c_name, amount):
        edge_index = g['ligand',c_name].edge_index
        edges_with_centers = torch.isin(edge_index[0], cluster_centers_idxs)
        edge_len = g['ligand'].pos[edge_index[0][edges_with_centers]] - g[c_name].pos[edge_index[1][edges_with_centers]]
        edge_len = torch.norm(edge_len, dim=1)
        groups = edge_index[0][edges_with_centers]
        clusters = []
        for idx in cluster_centers_idxs:
            curr_edge_len = edge_len[groups == idx]
            closest_in_cluster = torch.argsort(curr_edge_len)[:amount]
            idxs = edge_index[1][edges_with_centers][groups == idx][closest_in_cluster]
            node_vecs = g[c_name].x[idxs]
            edge_vecs = g['ligand',c_name].edge_attr[edges_with_centers][groups == idx][closest_in_cluster]
            center_vec = g['ligand'].x[idx].repeat(node_vecs.shape[0],1)
            cluster = torch.cat([node_vecs, edge_vecs, center_vec], dim=1)
            if cluster.shape[0] < amount:
                cluster = torch.cat([cluster, torch.zeros(amount - cluster.shape[0], cluster.shape[1], device=cluster.device)], dim=0)
            clusters.append(cluster)
        return clusters
        
    def _collect_local_clusters(self, graph_data, cluster_centers_idxs):
        if not self.add_hole_heighbours:
            clusters = graph_data['ligand'].x[cluster_centers_idxs]
            return clusters.unsqueeze(1)

        lig_clusters = self._get_local_clusters_vecs(graph_data, cluster_centers_idxs, 'ligand', 10)
        rec_clusters = self._get_local_clusters_vecs(graph_data, cluster_centers_idxs, 'receptor', 30)
        atom_clusters = self._get_local_clusters_vecs(graph_data, cluster_centers_idxs, 'atom', 20)
        clusters = []
        for l,r,c in zip(lig_clusters, rec_clusters, atom_clusters):
            clusters.append(torch.cat([l,r,c], dim=0).unsqueeze(0))
        return torch.cat(clusters)
        

    def _create_graph_memory(self, graph_data, molecule_sidechain_mask_idx):
        if self.graph_encoder is None:
            return None, None

        if not self.use_receptors:
            del graph_data['receptor']
            del graph_data['receptor','receptor']
            del graph_data['atom','receptor']
            del graph_data['ligand','receptor']

        neighbor_idxs = graph_data.hole_neighbors + graph_data["ligand"].ptr[
            :-1
        ].repeat_interleave(graph_data.num_sidechains)
        neighbor_idxs = self.graph_encoder.get_new_indexes_after_masking(
            graph_data, neighbor_idxs, molecule_sidechain_mask_idx
        )
        masked_graph_data = self.graph_encoder.mask_graph_sidechains(
            graph_data, molecule_sidechain_mask_idx
        )
        encoded_graph = self.graph_encoder(masked_graph_data, keep_hetrograph=True)
        graph_memory = self._collect_local_clusters(encoded_graph, neighbor_idxs)
        return graph_memory, torch.zeros(graph_memory.shape[0], graph_memory.shape[1]).bool().to(graph_memory.device)

    def _create_interaction_memory(self, interaction_data, num_sidechains):
        if self.interaction_encoder is None:
            return None, None
        interaction_data = list(interaction_data)
        interaction_data[0] = sum([[d]*n.item() for d,n in zip(interaction_data[0], num_sidechains)], [])
        interaction_data[1] = sum([[d]*n.item() for d,n in zip(interaction_data[1], num_sidechains)], [])
        interaction_memory = self.interaction_encoder(*interaction_data).unsqueeze(1)
        interaction_padding_mask = (
            torch.zeros(*interaction_memory.shape[0:2])
            .bool()
            .to(interaction_memory.device)
        )
        return interaction_memory, interaction_padding_mask

    def remove_nones(self, l):
        return [x for x in l if x is not None]

    def _create_memory(
        self,
        smiles_tokens_src,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
    ):

        smiles_memory, smiles_padding_mask = self._create_text_memory(smiles_tokens_src)
        graph_memory, graph_padding_mask = self._create_graph_memory(
            graph_data, molecule_sidechain_mask_idx
        )
        interaction_memory, interaction_padding_mask = self._create_interaction_memory(
            interaction_data, graph_data.num_sidechains
        )

        # Concatenate encoder output with GNN output
        combined_memory = torch.cat(
            self.remove_nones([smiles_memory, graph_memory, interaction_memory]), dim=1
        )
        memory_padding_mask = torch.cat(
            self.remove_nones([smiles_padding_mask, graph_padding_mask, interaction_padding_mask]), dim=1
        )
        return combined_memory, memory_padding_mask

    def generate_samples(
        self,
        num_samples,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
        **kwargs
    ):
        core_tokens = graph_data.core_tokens
        combined_memory, memory_padding_mask = self._create_memory(
            core_tokens, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        sidechains_lists = []
        for i in range(num_samples):
            batch_samples = self.decoder.generate(combined_memory, memory_padding_mask, **kwargs)
            batch_samples = batch_samples.cpu().numpy()
            sidechains_list = []
            splits = torch.cumsum(graph_data.num_sidechains, dim=0)
            for src, dst in zip([0,*splits[:-1]], [*splits]):
                sidechains_list.append(batch_samples[src:dst])
            sidechains_lists.append(sidechains_list)
        return sidechains_lists

    def optimized_generate_samples(
        self,
        num_samples,
        graph_data,
        interaction_data,
        tokenizer,
        molecule_sidechain_mask_idx=1,
        **kwargs
    ):
        core_tokens = graph_data.core_tokens
        combined_memory, memory_padding_mask = self._create_memory(
            core_tokens, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        sidechains_batches = []
        error_rate = 0
        for i in range(combined_memory.shape[0]):
            chains = []
            while len(chains) < num_samples:
                n = (num_samples - len(chains)) * (1 // (1.001 - error_rate))
                n = max(1,int(n * 1.2)) # increase the number of samples to account for errors
                memory = combined_memory[i].repeat(num_samples, 1,1)
                mask = memory_padding_mask[i].repeat(num_samples, 1)
                batch_samples = self.decoder.generate(memory, mask, **kwargs).cpu().numpy()
                gen_chains = tokenizer.decode_batch(batch_samples, skip_special_tokens=True)
                good_chains = [c for c in gen_chains if Chem.MolFromSmiles(f'[1*]{c}') is not None]
                error_rate = error_rate*0.9+  (1- len(good_chains) / len(gen_chains)) *0.1
                chains.extend(good_chains)
                if len(good_chains) == 0:
                    break
            sidechains_batches.append(chains[:num_samples]) 
        
        mols_sidechains_batches = []
        splits = torch.cumsum(graph_data.num_sidechains, dim=0)
        for src, dst in zip([0,*splits[:-1]], [*splits]):
            mols_sidechains_batches.append(sidechains_batches[src:dst])
        return mols_sidechains_batches        

    def forward(
        self,
        smiles_tokens_src,
        smiles_tokens_tgt,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
    ):

        combined_memory, memory_padding_mask = self._create_memory(
            smiles_tokens_src, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        # Transformer Decoder
        # if self.training:
        output = self._train_decode(
            smiles_tokens_tgt, combined_memory, memory_padding_mask
        )
        # output = self.decoder(smiles_tokens_tgt, combined_memory)
        return output
