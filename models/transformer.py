import torch
import torch.nn as nn
import math
from datetime import datetime




class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, tokenizer, embedding_dim, hidden_size, nhead, n_layers, max_length, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        vocab_dim = len(tokenizer.get_vocab())
        pad_token = tokenizer.token_to_id("<pad>")
        
        self.seq_len = max_length
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=self.seq_len)
        self.token_embedding = nn.Embedding(vocab_dim, embedding_dim)
        enc_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_size, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pad_token = pad_token

    def create_src_key_padding_mask(self, src):
        return (src == self.pad_token)

    def forward(self, src, src_key_padding_mask):
        
        embedded_src = self.token_embedding(src.permute(1, 0))
        final_src = self.positional_encoding(embedded_src)
        output = self.encoder(final_src.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
        return output



class TransformerDecoder(nn.Module):
    def __init__(self, tokenizer, embedding_dim, hidden_size, nhead, n_layers, max_length,
                 dropout=0.1, ):
        super(TransformerDecoder, self).__init__()
        
        pad_token=tokenizer.token_to_id("<pad>")
        start_token=tokenizer.token_to_id("<bos>")
        end_token=tokenizer.token_to_id("<eos>")
        vocab_dim = len(tokenizer.get_vocab())

        self.seq_len = max_length
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=self.seq_len)
        self.token_embedding = nn.Embedding(vocab_dim, embedding_dim)
        dec_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, hidden_size, dropout, batch_first=True)
        self.decoder : nn.TransformerDecoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.dense = nn.Linear(embedding_dim, vocab_dim)
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
    
    @property
    def device(self):
        return next(self.parameters()).device

    def _generate_square_subsequent_mask(self, sz, device):
        return (torch.triu(torch.ones((sz, sz), device=device), diagonal=0) == 1)
        
        # mask = (torch.triu(torch.ones((sz, sz), device=device), diagonal=0) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # return mask


    def create_target_masks(self, tgt):
        tgt_seq_len = tgt.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, tgt.device)
        tgt_padding_mask = (tgt == self.pad_token)
        return tgt_mask, tgt_padding_mask

    def forward(self, target, memory, target_mask, target_padding_mask,memory_key_padding_mask=None, memory_mask=None):
        embedded_target = self.token_embedding(target.permute(1, 0))
        final_target = self.positional_encoding(embedded_target)
        output = self.decoder(final_target.permute(1, 0, 2), memory,
                              tgt_mask=target_mask, tgt_key_padding_mask=target_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask, memory_mask=memory_mask)
        logits = self.dense(output)
        return logits

    def generate(self, cond_memory, memory_key_padding_mask=None, memory_mask=None, max_len=None,
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
            out = self(ys, cond_memory, target_mask=None,target_padding_mask=None, memory_key_padding_mask=memory_key_padding_mask, memory_mask=memory_mask)
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
