import pickle
import gzip
import scipy.stats as sps
import numpy as np
import os.path

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import sascorer
import networkx as nx

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

from jtnn import *

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from optparse import OptionParser


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# We define the functions used to load and save objects
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret

parser = OptionParser()
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-o", "--save_dir", dest="save_dir")
parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-r", "--seed", dest="random_seed", default=None)
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
random_seed = int(opts.random_seed)

model = DGLJTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = cuda(model)

# We load the random seed
np.random.seed(random_seed)

# We load the data (y is minued!)
X = np.loadtxt('./bo/latent_features2.txt')
y = -np.loadtxt('./bo/targets2.txt')
y = y.reshape((-1, 1))

n = X.shape[ 0 ]

permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

y_train = y_train.transpose()
y_test = y_test.transpose()

np.random.seed(random_seed)

logP_values = np.loadtxt('./bo/logP_values2.txt')
SA_scores = np.loadtxt('./bo/SA_scores2.txt')
cycle_scores = np.loadtxt('./bo/cycle_scores2.txt')
SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)


M = 500 #num of inducing points
# initialize the inducing inputs
lb=np.min(X_train, 0)
ub=np.max(X_train,0)
Xu = np.random.rand(M,X_train.shape[1])
Xu=Xu*(ub-lb)+lb

# initialize the kernel and model
kernel = gp.kernels.RBF(input_dim=X_train.shape[1])

X_train=torch.from_numpy(X_train).float()
y_train=torch.from_numpy(y_train).float()
X_test=torch.from_numpy(X_test).float()
y_test=torch.from_numpy(y_test).float()

print(X_train.shape)
Xu=torch.from_numpy(Xu).float()
lb=torch.from_numpy(lb).float()
ub=torch.from_numpy(ub).float()

# we increase the jitter for better numerical stability
gpmodel = gp.models.SparseGPRegression(X_train, y_train, kernel, Xu=Xu, jitter=1.0e-3)
optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
gp.util.train(gpmodel, optimizer)
pred, _ = gpmodel(X_train, full_cov=False, noiseless=False)
RMSE = np.sqrt(np.mean((pred.detach().numpy() - y_train.numpy())**2))
print("Train RMSE: ",RMSE)
pred, _ = gpmodel(X_test, full_cov=False, noiseless=False)
RMSE = np.sqrt(np.mean((pred.detach().numpy() - y_test.numpy())**2))
print("Test RMSE: ",RMSE)

def update_posterior(X_new, Y_new):

    gpmodel.set_data(X_new, Y_new)
    # optimize the GP hyperparameters using Adam with lr=0.001
    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
    gp.util.train(gpmodel, optimizer)

def lower_confidence_bound(x, kappa=2):
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    sigma = variance.sqrt()
    return mu - kappa * sigma

def find_a_candidate(x_init, lower_bound, upper_bound):
    # transform x to an unconstrained domain
    constraint = constraints.interval(lower_bound, upper_bound)
    #print(x_init)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    #print(unconstrained_x_init)
    unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
    minimizer = optim.LBFGS([unconstrained_x])

    def closure():
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        y = lower_confidence_bound(x)
        autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
        return y

    minimizer.step(closure)
    # after finding a candidate in the unconstrained domain,
    # convert it back to original domain.
    x = transform_to(constraint)(unconstrained_x)
    return x.detach()

def next_x(lower_bound, upper_bound, num_candidates_each_x=5, num_x=60):
    found_x=[]
    x_init = gpmodel.X[-1:]
    for j in range(num_x):
        for i in range(num_candidates_each_x):
            candidates = []
            values = []
            x = find_a_candidate(x_init, lower_bound, upper_bound)
            #print("next",x)
            y = lower_confidence_bound(x)
            candidates.append(x)
            values.append(y)
            x_init = x.new_empty(1,56).uniform_(0, 1).mul(ub-lb).add_(lb)
            #print("next",x_init)
        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        found_x.append(candidates[argmin])
        x_init=found_x[-1]
    return found_x

#BO 5 times
max_iteration=5
for iteration in range(max_iteration):

    pyro.set_rng_seed((iteration+1) * random_seed)

    xmin = next_x(lb,ub,5,60)
    valid_smiles=[]
    scores=[]
    for x_new in xmin:
        #model = DGLJTNNVAE(vocab, hidden_size, latent_size, depth)
        #model.load_state_dict(torch.load(opts.model_path))
        #model = cuda(model)
        tree_vec, mol_vec = x_new.chunk(2,1)
        print(x_new.shape, tree_vec.shape, mol_vec.shape)
        print(x_new)
        s=model.decode(tree_vec, mol_vec)
        if s is not None:
            valid_smiles.append(s)

            current_log_P_value = Descriptors.MolLogP(MolFromSmiles(s))
            current_SA_score = -sascorer.calculateScore(MolFromSmiles(s))
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(s))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([ len(j) for j in cycle_list ])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6

            current_cycle_score = -cycle_length

            current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
            current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
            current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

            score = current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized
            y_new=-score
            scores.append(y_new)
            
            #print(gpmodel.X.shape)
            #print(x_new.shape)

            X = torch.cat((gpmodel.X, x_new),0) # incorporate new evaluation
            #print(X.shape)
            #print(gpmodel.y.shape)
            #print(y_new)
            #print(torch.tensor(y_new).float())
            #print(torch.tensor(y_new).float().shape)
            #print(torch.tensor([[y_new]]).float())
            y = torch.cat((gpmodel.y, torch.tensor([[y_new]]).float()),1)
            
            #print(y.shape)
            #print(y.type())
    if iteration < max_iteration-1:
        update_posterior(X, y)
        pred, _ = gpmodel(X_train, full_cov=False, noiseless=False)
        RMSE = np.sqrt(np.mean((pred.detach().numpy() - y_train.numpy())**2))
        print("Train RMSE: ",RMSE)
        pred, _ = gpmodel(X_test, full_cov=False, noiseless=False)
        RMSE = np.sqrt(np.mean((pred.detach().numpy() - y_test.numpy())**2))
        print("Test RMSE: ",RMSE)
    print(len(scores)," new molecules are found. Iteration-",iteration)
    print(valid_smiles)
    print(scores)
    save_object(valid_smiles, opts.save_dir + "/valid_smiles{}.txt".format(iteration))
    save_object(scores, opts.save_dir + "/scores{}.txt".format(iteration))

