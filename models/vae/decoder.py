import torch
from models.transformer import TransformerDecoder


class VaeDecoder(TransformerDecoder):
    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        self.bos_lin = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
    
    def forward(self, bos_emb, target, memory, target_mask, target_padding_mask,memory_key_padding_mask=None, memory_mask=None):
        embedded_target = self.token_embedding(target.permute(1, 0))
        embedded_target[0] = self.bos_lin(bos_emb)
        final_target = self.positional_encoding(embedded_target)
        output = self.decoder(final_target.permute(1, 0, 2), memory,
                              tgt_mask=target_mask, tgt_key_padding_mask=target_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask, memory_mask=memory_mask)
        logits = self.dense(output)
        if self.return_token_embeddings:
            return logits, output
        return logits

    def generate(self, bos, cond_memory, memory_key_padding_mask=None, memory_mask=None, max_len=None,
                 k=None, p=None, prefix=None):
        assert (k is not None) != (p is not None), "must use only one option, topp or topk"
        batch_size = cond_memory.shape[0]
        max_len = max_len or self.seq_len
        device = cond_memory.device
        if prefix is None:
            ys = torch.ones(batch_size, 1).type(torch.int).fill_(self.start_token).to(device)
            generation_length = max_len - 1
        else:
            ys = torch.tensor(prefix).reshape(batch_size, -1).to(device)
            generation_length = max_len - len(prefix)
        finished_sample = torch.zeros(batch_size).bool().to(device)
        for i in range(generation_length):
            out = self(bos, ys, cond_memory, target_mask=None,target_padding_mask=None, memory_key_padding_mask=memory_key_padding_mask, memory_mask=memory_mask)
            if self.return_token_embeddings:
                out, _ = out
            out = out / 0.5
            probs = torch.softmax(out, dim=2) 
            sorted_probs, sorted_indices = torch.sort(probs[:, -1, :], descending=True)
            if k is not None:
                sorted_probs[:,k:] = 0
            else:
                cum_probs = torch.cumsum(sorted_probs, dim=1)
                top_p_mask = cum_probs > p
                if torch.all(top_p_mask):
                    top_p_mask[:,0] = False
                sorted_probs[top_p_mask] = 0
            remainder_sum = torch.sum(sorted_probs, 1)
            sorted_probs = sorted_probs / remainder_sum.unsqueeze(-1)
            next_sorted_token = torch.multinomial(sorted_probs, 1).squeeze(-1)
            next_token_id = sorted_indices[torch.arange(sorted_indices.shape[0]), next_sorted_token]
            next_token_id[finished_sample] = self.pad_token
            ys = torch.cat([ys,next_token_id.unsqueeze(-1)], dim=1) 
            finished_sample = finished_sample | (next_token_id == self.end_token)
            if finished_sample.all() or ys.shape[1] == max_len:
                break
        return ys
    