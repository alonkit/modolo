
import torch


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims, lm_embedding_dim=0):
        """

        Parameters
        ----------
        emb_dim
        feature_dims
            first element of feature_dims tuple is a list with the length of each categorical feature,
            and the second is the number of scalar features
        sigma_embed_dim
        lm_embedding_dim
        """
        #
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.additional_features_dim = feature_dims[1] + lm_embedding_dim
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.additional_features_dim > 0:
            self.additional_features_embedder = torch.nn.Linear(self.additional_features_dim + emb_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.additional_features_dim
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.additional_features_dim > 0:
            x_embedding = self.additional_features_embedder(torch.cat([x_embedding, x[:, self.num_categorical_features:]], axis=1))
        return x_embedding