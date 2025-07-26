import numpy as np
import prody as pr
from datasets.process_chem.parse_chi import aa_idx2aa_short, get_onehot_sequence


def get_sequences_from_protein(pdb):
    sequence = None
    if isinstance(pdb,str):
        pdb = pr.parsePDB(pdb)
    seq = pdb.ca.getSequence()
    one_hot = get_onehot_sequence(seq)

    chain_ids = np.zeros(len(one_hot))
    res_chain_ids = pdb.ca.getChids()
    res_seg_ids = pdb.ca.getSegnames()
    res_chain_ids = np.asarray([s + c for s, c in zip(res_seg_ids, res_chain_ids)])
    ids = np.unique(res_chain_ids)

    for i, id in enumerate(ids):
        chain_ids[res_chain_ids == id] = i

        s_temp = np.argmax(one_hot[res_chain_ids == id], axis=1)
        s = ''.join([aa_idx2aa_short[aa_idx] for aa_idx in s_temp])

        if sequence is None:
            sequence = s
        else:
            sequence += (":" + s)

    return sequence
