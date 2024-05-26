import numpy as np
from tqdm import tqdm
import argparse

import gudhi
import torch
import numpy as np
import torch
import networkx as nx
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from model.scattering import scattering_transform
from preprocessing.simplicial_construction import _get_transition_matrix, _get_laplacians, _get_simplex_features_gc, get_boundary_matrices_from_processed_tree
from preprocessing.graph_construction import read_graph_data
from model.model import LogReg

import warnings
warnings.filterwarnings("ignore")

import gc
gc.enable()

parser = argparse.ArgumentParser(description='SSN')

parser.add_argument('--data', type=str, default='proteins', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--J', type=int, default=2, help='Maximum scale')
parser.add_argument('--include_boundary', type=bool,  default=True, help='If boundary information should be included or not')

args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    print(args)
    labels = np.load('data/graph classification/'+args.data+'/label_sets_'+args.data+'.npy', allow_pickle=True)
    simplicial = np.load('data/graph classification/'+args.data+'/simplicial_sets_'+args.data+'.npy', allow_pickle=True)

    SCs, INDs, _labels, simplex_trees, node_attributes, netxG = read_graph_data(args.data, labels, simplicial)
    labels = np.array(_labels)
    index_adj = {'B':0, 'C':1, 'L':2, 'U':3}
    keep = []
    features = []
    print("Extracting features....")
    for graph in tqdm(range(len(labels))):
            simplex_tree = simplex_trees[graph]
            sc = SCs[graph]
            X = torch.FloatTensor(node_attributes[graph]).to(args.device)
            index = INDs[graph]
            if(args.data=='nci1' or args.data=='enzymes'):
                _, sc, boundaries = get_boundary_matrices_from_processed_tree(simplex_tree,sc,index,1)
            else:
                _, sc, boundaries = get_boundary_matrices_from_processed_tree(simplex_tree,sc,index,3)
            X = _get_simplex_features_gc(sc[1:], X)
            lower_laplacians, upper_laplacians = _get_laplacians(boundaries)
            P_B, P_L, P_U = _get_transition_matrix(boundaries, lower_laplacians, upper_laplacians)
            try:
                Psi = scattering_transform([i.view(len(i),1) for i in X], P_B, P_L, P_U, index_adj, args.J, args.include_boundary)
                Psi_Psi = []
                for PsiX in Psi:
                    Psi_Psi.append(scattering_transform(PsiX, P_B, P_L, P_U, index_adj, args.J, args.include_boundary))
                Phi = []
                for k in range(len(X)):
                    Phi_k = X[k].view(len(X[k]),1)
                    for j in range(args.J):
                        Phi_k = torch.cat((Phi_k, Psi[j][k]),axis=1)
                    for j in range(args.J):
                        for _j in range(args.J):
                            Phi_k = torch.cat((Phi_k, Psi_Psi[j][_j][k]),axis=1)
                    Phi.append(Phi_k)
                x = []
                for phi in Phi:
                    x.append(phi.sum(0))
                features.append(torch.cat(x).cpu().detach().numpy())
                keep.append(graph)
            except RuntimeError:
                continue
    X = np.array(features)
    X = MinMaxScaler().fit_transform(X)
    y = np.array([labels[i]  for i in keep])
    
    print(f"New length of dataset: {len(y)}")

    kf = KFold(n_splits=10, shuffle=False)

    print("Training Logistic Regression")
    acc = []
    for i, (train_index, test_index) in tqdm(enumerate(kf.split(X, y))):
        X_train = torch.FloatTensor(X[train_index]).to(args.device)
        y_train = torch.LongTensor(y[train_index]).to(args.device)
        X_test = torch.FloatTensor(X[test_index]).to(args.device)
        y_test = torch.LongTensor(y[test_index]).to(args.device)

        for repeat in range(3):
            model = LogReg(X_train.shape[1], len(np.unique(y)))
            opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0)
            # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = 5000, gamma=0.001, verbose=False)
            model = model.to(args.device)
            loss_fn = torch.nn.CrossEntropyLoss()
            best_acc = 0
            for epoch in range(10000):
                model.train()
                opt.zero_grad()
                logits = model(X_train)
                preds = torch.argmax(logits, dim=1)
                train_acc = torch.sum(preds == y_train).float() / y_train.shape[0]
                loss = loss_fn(logits, y_train)
                loss.backward()
                opt.step()
                model.eval()
                # scheduler.step()
                with torch.no_grad():
                    test_logits = model(X_test)
                    test_preds = torch.argmax(test_logits, dim=1)
                    test_acc = torch.sum(test_preds == y_test).float() / y_test.shape[0]

                    if test_acc > best_acc:
                        best_acc = test_acc
            acc.append(best_acc.item())
    acc = np.array(acc)*100
    print(f"Accuracy = {acc.mean()}, Standard deviation = {acc.std()}")
