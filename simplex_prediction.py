import argparse
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch

from preprocessing.graph_construction import _get_graph
from preprocessing.simplicial_construction import get_boundary_matrices_sp,_get_laplacians,_get_simplex_features, _get_transition_matrix
from model.scattering import scattering_transform
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SSN')

parser.add_argument('--data', type=str, default='madison-restaurant-reviews', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--dim', type=int, default=10, help='Order of the simplicial complex.')
parser.add_argument('--J', type=int, default=5, help='Maximum scale')
parser.add_argument('--split', type=float, default=0.2, help='Test data size')
parser.add_argument('--include_boundary', type=bool,  default=True, help='If boundary information should be included or not')

args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


if __name__ == '__main__':
    print(args)
    simplex_tree, sc, boundry_matrices, labels, simplicies =  get_boundary_matrices_sp(args.data, args.dim)
    g, netxG = _get_graph(sc[1], num_nodes=sc[0].max()+1)
    netxG.add_nodes_from(np.setdiff1d(np.arange(0,max(list(netxG.nodes()))), np.array(list(netxG.nodes()))))
    g = g.to(args.device)
    try:
        g.ndata['features'] = torch.load('features/'+args.data+'.th').to(args.device)
    except FileNotFoundError:
        pass
    lower_laplacians, upper_laplacians = _get_laplacians(boundry_matrices)
    P_B, P_L, P_U = _get_transition_matrix(boundry_matrices, lower_laplacians, upper_laplacians)
    index = {'B':0, 'C':1, 'L':2, 'U':3}

    X = _get_simplex_features(sc[1:], g.ndata['features'])

    Psi = scattering_transform(X, P_B, P_L, P_U, index, args.J, args.include_boundary)
    Psi_Psi = []
    for PsiX in Psi:
        Psi_Psi.append(scattering_transform(PsiX, P_B, P_L, P_U, index, args.J, args.include_boundary))

    Phi = []
    for k in range(len(X)):
        Phi_k = X[k]
        for j in range(args.J):
            Phi_k = torch.cat((Phi_k, Psi[j][k]),axis=1)
        for j in range(args.J):
            for _j in range(args.J):
                Phi_k = torch.cat((Phi_k, Psi_Psi[j][_j][k]),axis=1)
            Phi.append(Phi_k)        
    X = Phi[0].cpu().detach().numpy()
    new_x = []
    for i in simplicies:
        new_x.append(X[np.array(i)-1].mean(0))
    X = np.array(new_x)
    y = labels
    acc = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = args.split, stratify=labels)
        classifier = RidgeClassifier()
        classifier.fit(X_train, y_train)
        acc.append(accuracy_score(y_test, classifier.predict(X_test))*100)
    acc = np.array(acc)
    print(f"Accuracy = {acc.mean()}, Standard deviation = {acc.std()}")