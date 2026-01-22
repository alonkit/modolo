

import torch
from models.modolo import Modolo
from models.vae.cvae import CVAE
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

class CfomDockVae(Modolo):
    
    def __init__(self, use_cvae=True,*args,**kws):
        super().__init__(*args,**kws)    
        self.memory_scaler = torch.nn.Linear(
            self.graph_encoder.edge_channels+self.graph_encoder.out_channels, 
            self.graph_encoder.out_channels)
        
        self.decoder_input_dropout = torch.nn.Dropout(0.3)
        self.use_cvae = use_cvae
        if use_cvae:
            self.cvae = CVAE(
                d_model=self.graph_encoder.out_channels,
                d_latent=32,
            )
    
    def _create_graph_memory(self, graph_data, molecule_sidechain_mask_idx):
        if self.graph_encoder is None:
            return None, None

        if not self.use_receptors:
            del graph_data['receptor']
            del graph_data['receptor','receptor']
            del graph_data['atom','receptor']
            del graph_data['ligand','receptor']
            del graph_data['frag','receptor']

        graph_data = self.prep_graph(graph_data)
        encoded_graph = self.graph_encoder(graph_data, keep_hetrograph=True)
        graph_memory, graph_padding_mask = self.stack_ligand_memory(encoded_graph, graph_data)
        graph_memory = self.memory_scaler(graph_memory)
        # return encoded_graph['ligand'].x[graph_data['ligand'].frag_hole] ,graph_memory, graph_padding_mask.to(graph_memory.device)
        return encoded_graph['frag'].x ,graph_memory, graph_padding_mask.to(graph_memory.device)
    
    # def stack_ligand_memory(self, encoded_graph, graph_data):
    #     mems = []
    #     for hole_idx in graph_data['ligand'].frag_hole:
    #         curr_mems = []
    #         for tgt_type in ['ligand','receptor', 'atom']:
    #             ei = graph_data['ligand',tgt_type].edge_index
    #             relevant = ei[0]==hole_idx
    #             ei = ei[:,relevant]
    #             ea = encoded_graph['ligand',tgt_type].edge_attr[relevant]
    #             nds = encoded_graph[tgt_type].x[ei[1]]
    #             mem = torch.cat([ea, nds],1)
    #             curr_mems.append(mem)
    #         curr_mems = torch.cat(curr_mems,0)
    #         mems.append(curr_mems)
    #     lengths = [mem.shape[0] for mem in mems]
    #     mems = pad_sequence(mems, batch_first=True, padding_value=0.0) # T, B, D
    #     lengths = torch.tensor(lengths)
    #     arange_T = torch.arange(mems.shape[1])
    #     valid_data_mask = arange_T.unsqueeze(0) < lengths.unsqueeze(1)
    #     return mems, ~valid_data_mask

    def stack_ligand_memory(self, encoded_graph, graph_data):
        mems = []
        for hole_idx in range(graph_data['frag'].x.shape[0]):
            curr_mems = []
            tgts = ['ligand','receptor', 'atom'] if self.use_receptors else ['ligand','atom']
            for tgt_type in tgts:
                ei = graph_data['frag',tgt_type].edge_index
                relevant = ei[0]==hole_idx
                ei = ei[:,relevant]
                ea = encoded_graph['frag',tgt_type].edge_attr[relevant]
                nds = encoded_graph[tgt_type].x[ei[1]]
                mem = torch.cat([ea, nds],1)
                curr_mems.append(mem)
            curr_mems = torch.cat(curr_mems,0)
            mems.append(curr_mems)
        lengths = [mem.shape[0] for mem in mems]
        mems = pad_sequence(mems, batch_first=True, padding_value=0.0) # T, B, D
        lengths = torch.tensor(lengths)
        arange_T = torch.arange(mems.shape[1])
        valid_data_mask = arange_T.unsqueeze(0) < lengths.unsqueeze(1)
        return mems, ~valid_data_mask


    def _create_memory(
        self,
        smiles_tokens_src,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
    ):

        graph_tgt_start, graph_memory, graph_padding_mask = self._create_graph_memory(
            graph_data, molecule_sidechain_mask_idx
        )
        interaction_memory, interaction_padding_mask = self._create_interaction_memory(
            interaction_data, graph_data['ligand'].num_frags
        )

        # Concatenate encoder output with GNN output
        combined_memory = graph_memory
        memory_padding_mask = graph_padding_mask
        return graph_tgt_start + interaction_memory.squeeze(1), combined_memory, memory_padding_mask

    def optimized_generate_samples(
        self,
        num_samples,
        graph_data,
        interaction_data,
        tokenizer,
        molecule_sidechain_mask_idx=1,
        **kwargs
    ):
        bos, combined_memory, memory_padding_mask = self._create_memory(
            None, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        
        sidechains_batches = []
        error_rate = 0
        for i in range(combined_memory.shape[0]):
            chains = []
            while len(chains) < num_samples:
                n = (num_samples - len(chains)) * (1 // (1.001 - error_rate))
                n = max(1,int(n * 1.2)) # increase the number of samples to account for errors
                memory = combined_memory[i].repeat(num_samples, 1,1)
                c_bos = bos[i].repeat(num_samples,1)
                mask = memory_padding_mask[i].repeat(num_samples, 1)
                if self.use_cvae:
                    c_bos, kld_loss = self.cvae(
                        c_bos, None, training=False
                    )
                else:
                    kld_loss = torch.tensor(0.0, device=c_bos.device)
                batch_samples = self.decoder.generate(c_bos, memory, mask, **kwargs).cpu().numpy()
                gen_chains = tokenizer.decode_batch(batch_samples, skip_special_tokens=True)
                good_chains = [c for c in gen_chains if Chem.MolFromSmiles(f'[1*]{c}') is not None]
                error_rate = error_rate*0.9+  (1- len(good_chains) / len(gen_chains)) *0.1
                chains.extend(good_chains)
                if len(good_chains) == 0:
                    break
            sidechains_batches.append(chains[:num_samples]) 
        
        mols_sidechains_batches = []
        splits = torch.cumsum(graph_data['ligand'].num_frags, dim=0)
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

        
        graph_tgt_start, combined_memory, memory_padding_mask = self._create_memory(
            smiles_tokens_src, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        
        
        if self.use_cvae:
            smiles_memory, smiles_padding_mask = self._create_text_memory(smiles_tokens_tgt)
            fragment_embedding = self.masked_mean_pooling(
                smiles_memory, ~smiles_padding_mask        
            )
            
            graph_tgt_start, kld_loss = self.cvae(
                graph_tgt_start, fragment_embedding, training=True
            )
        else:
            kld_loss = torch.tensor(0.0, device=graph_tgt_start.device)
        
            
        output = self._train_decode(
            self.decoder_input_dropout(graph_tgt_start), 
            smiles_tokens_tgt, 
            self.decoder_input_dropout(combined_memory), 
            memory_padding_mask
        )
        return output, kld_loss

    @staticmethod
    def masked_mean_pooling(embeddings, mask):
        """
        embeddings: (B, L, D) - Output from your Transformer/Embedding layer
        mask: (B, L) - 1 for valid tokens, 0 for padding
        """
        
        # 1. Expand mask to match embedding dimensions (B, L) -> (B, L, 1)
        mask_expanded = mask.unsqueeze(-1).float()
        
        # 2. Sum the embeddings (mask zeros out the padding positions)
        #    We multiply first to ensure padding vectors are strictly 0.0
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        
        # 3. Count how many valid tokens are in each batch item
        #    Result shape: (B, 1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        
        # 4. Divide (Clamp count to avoid division by zero error for empty sequences)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings # Shape: (B, D)

    def _train_decode(self, bos_emb, tgt, memory, memory_key_padding_mask):
        target_mask, target_padding_mask = self.decoder.create_target_masks(tgt)
        output = self.decoder(
            bos_emb,
            tgt,
            memory,
            target_mask,
            target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output