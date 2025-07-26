from esm import FastaBatchedDataset, pretrained
import torch
from tqdm import tqdm

def compute_ESM_embeddings(model, alphabet, labels, sequences):
        # settings used
        toks_per_batch = 4096
        repr_layers = [33]
        include = "per_tok"
        truncation_seq_length = 1022

        dataset = FastaBatchedDataset(labels, sequences)
        batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
        )

        assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
        repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
        embeddings = {}

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)

                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

                for i, label in enumerate(labels):
                    truncate_len = min(truncation_seq_length, len(strs[i]))
                    embeddings[label] = representations[33][i, 1: truncate_len + 1].clone()
        return embeddings
