import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
from optparse import OptionParser
from collections import deque
import rdkit

from jtnn import *

torch.multiprocessing.set_sharing_strategy('file_system')

def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
worker_init_fn(None)

parser = OptionParser()
parser.add_option("-t", "--train", dest="train", default='train', help='Training file name')
parser.add_option("-v", "--vocab", dest="vocab", default='vocab', help='Vocab file name')
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-m", "--model", dest="model_path", default=None)
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-z", "--beta", dest="beta", default=0.001)
parser.add_option("-q", "--lr", dest="lr", default=1e-3)
parser.add_option("-T", "--test", dest="test", action="store_true")
opts,args = parser.parse_args()

dataset = JTNNDataset(data=opts.train, vocab=opts.vocab, training=True)
vocab = dataset.vocab
batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
lr = float(opts.lr)

model = DGLJTNNVAE(vocab, hidden_size, latent_size, depth)

if opts.model_path is not None:
    model.load_state_dict(torch.load(opts.model_path))
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)

model = cuda(model)
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

MAX_EPOCH = 100
PRINT_ITER = 20

def train():
    dataset.training = True
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=JTNNCollator(vocab, True),
            drop_last=True,
            worker_init_fn=worker_init_fn)

    for epoch in range(MAX_EPOCH):
        word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0

        for it, batch in enumerate(dataloader):
            model.zero_grad()
            try:
                loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta)
            except:
                print([t.smiles for t in batch['mol_trees']])
                raise
            loss.backward()
            optimizer.step()

            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc

            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100

                print("KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Loss: %.6f" % (
                    kl_div, word_acc, topo_acc, assm_acc, steo_acc, loss.item()),'  -',epoch,' ',it+1)
                word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0
                sys.stdout.flush()
            if epoch==0 and it==2:
                torch.save(model.state_dict(),opts.save_path+"/model.iter-%d-%d-e7" % (epoch, it + 1))
            if (it + 1) % 300 == 0: #Fast annealing
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
                torch.save(model.state_dict(),
                           opts.save_path + "/model.iter-%d-%d-e7" % (epoch, it + 1))

        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))

def test():
    print('testing')
    dataset.training = False
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=JTNNCollator(vocab, False),
            drop_last=True,
            worker_init_fn=worker_init_fn)

    # Just an example of molecule decoding; in reality you may want to sample
    # tree and molecule vectors.
    total=0
    reconstruct=0
    for it, batch in enumerate(dataloader):
        gt_smiles = batch['mol_trees'][0].smiles
        print(gt_smiles)

        model.move_to_cuda(batch)
        _, tree_vec, mol_vec = model.encode(batch)
        for decode_num in range(10):
            sample_tree_vec, sample_mol_vec, _, _ = model.sample(tree_vec, mol_vec)
            #print(sample_tree_vec,sample_mol_vec)
            #print(sample_tree_vec.shape,sample_mol_vec.shape)
            smiles = model.decode(sample_tree_vec, sample_mol_vec)
            print(smiles,' :: ',decode_num)
            if gt_smiles==smiles:
                reconstruct=reconstruct+1
            total=total+1
        if (it+1)%5==0:
            print('accuracy:%.6f'%(reconstruct*1.0/total))

if __name__ == '__main__':
    if opts.test:
        test()
    else:
        train()

    print('# passes:', model.n_passes)
    print('Total # nodes processed:', model.n_nodes_total)
    print('Total # edges processed:', model.n_edges_total)
    print('Total # tree nodes processed:', model.n_tree_nodes_total)
    print('Graph decoder: # passes:', model.jtmpn.n_passes)
    print('Graph decoder: Total # candidates processed:', model.jtmpn.n_samples_total)
    print('Graph decoder: Total # nodes processed:', model.jtmpn.n_nodes_total)
    print('Graph decoder: Total # edges processed:', model.jtmpn.n_edges_total)
    print('Graph encoder: # passes:', model.mpn.n_passes)
    print('Graph encoder: Total # candidates processed:', model.mpn.n_samples_total)
    print('Graph encoder: Total # nodes processed:', model.mpn.n_nodes_total)
    print('Graph encoder: Total # edges processed:', model.mpn.n_edges_total)
