import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import dgl
from dgl.data.utils import download, extract_archive, get_download_dir
from torch.utils.data import DataLoader
from optparse import OptionParser

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import sascorer
import networkx as nx

import numpy as np
from jtnn import *

def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
worker_init_fn(None)

parser = OptionParser()
parser.add_option("-t", "--train", dest="train", default='train', help='Training file name')
parser.add_option("-v", "--vocab", dest="vocab", default='vocab', help='Vocab file name')
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
opts,args = parser.parse_args()

dataset = JTNNDataset(data=opts.train, vocab=opts.vocab, training=False)
vocab = dataset.vocab
smiles=dataset.data

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)

model = DGLJTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = cuda(model)

smiles_rdkit = []
for i in range(len(smiles)):
    smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[ i ]), isomericSmiles=True))
    if i%10000==0:
        print("smile,",i)
logP_values = []
for i in range(len(smiles)):
    logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))
    if i%10000==0:
        print("logP,",i)
SA_scores = []
for i in range(len(smiles)):
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[ i ])))
    if i%10000==0:
        print("sa,",i)
cycle_scores = []
for i in range(len(smiles)):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)
    if i%10000==0:
        print("cycle,",i)
SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

latent_points = []
dataloader = DataLoader(
        dataset,
        batch_size=40,
        shuffle=False,
        num_workers=0,
        collate_fn=JTNNCollator(vocab, False),
        drop_last=True,
        worker_init_fn=worker_init_fn)

for it, batch in enumerate(dataloader):
    model.move_to_cuda(batch)
    _, tree_vec, mol_vec = model.encode(batch)
    _, _, z_mean, _ = model.sample(tree_vec, mol_vec)
    latent_points.append(z_mean.data.cpu().numpy())
    if it%20==0:
        print("latent,",it)
latent_points = np.vstack(latent_points)
np.savetxt('bo/latent_features2.txt', latent_points)
targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
np.savetxt('bo/targets2.txt', targets)
np.savetxt('bo/logP_values2.txt', np.array(logP_values))
np.savetxt('bo/SA_scores2.txt', np.array(SA_scores))
np.savetxt('bo/cycle_scores2.txt', np.array(cycle_scores))
