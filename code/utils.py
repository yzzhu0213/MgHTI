import dgl
from rdkit import Chem
import numpy as np
import torch


def load_data():
    herb_init = np.load('input/herb_init_feature.npy', allow_pickle=True)
    target_init = np.load('input/target_init_feature.npy', allow_pickle=True)
    ingredient_init = np.load("input/ingredient_init_feature.npy")
    instance_idx = np.load('input/instance_idx.npy', allow_pickle=True)
    labels = np.load('input/labels.npy', allow_pickle=True)
    train_idx = np.load('input/train_idx.npy', allow_pickle=True)
    val_idx = np.load('input/val_idx.npy', allow_pickle=True)
    test_idx = np.load('input/test_idx.npy', allow_pickle=True)

    return (
        herb_init,
        target_init,
        ingredient_init,
        instance_idx,
        labels,
        train_idx,
        val_idx,
        test_idx,
    )


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smiles_to_dgl(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string")

    mol = Chem.AddHs(mol)

    g = dgl.DGLGraph()

    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    g = dgl.add_self_loop(g)

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edges(start, end)
        g.add_edges(end, start)

    atom_feats = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        atom_feats.append(feature / sum(feature))
    g.ndata['feat'] = torch.tensor(atom_feats, dtype=torch.float32)

    return g


def seq_code(prot):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    max_seq_len = 500
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

