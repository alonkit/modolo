import torch

import torch.nn as nn

class InteractionEncoder(nn.Module):
    INTERACTION_TYPE = {'B': 0, 'F': 1, 'A': 2}
    UNKNOWN_TYPE = 3

    def __init__(self, output_dim):
        super(InteractionEncoder, self).__init__()
        self.embedding = nn.Embedding((len(self.INTERACTION_TYPE) + 1)*2, output_dim)

    def forward(self, interaction_types, labels):
        emb_ids = [self.INTERACTION_TYPE.get(t, self.UNKNOWN_TYPE)*2+int(l) for t, l in zip(interaction_types, labels)]
        emb_ids = torch.tensor(emb_ids, dtype=torch.long).to(self.embedding.weight.device)
        return self.embedding(emb_ids)

