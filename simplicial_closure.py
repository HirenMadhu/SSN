import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, auc, precision_recall_curve

import torch

from preprocessing.graph_construction import _get_graph
from preprocessing.simplicial_construction import get_boundary_matrices,_get_laplacians,_get_simplex_features, _get_transition_matrix, get_simplicies_closure, get_boundary_matrices_from_processed_tree
from model.scattering import scattering_transform
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SSN')

parser.add_argument('--data', type=str, default='contact-high-school', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--dim', type=int, default=3, help='Order of the simplicial complex.')
parser.add_argument('--J', type=int, default=4, help='Maximum scale')
parser.add_argument('--split', type=float, default=0.2, help='Test data size')
parser.add_argument('--include_boundary', type=bool,  default=True, help='If boundary information should be included or not')

args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

rb = {'contact-high-school':0.0112, 'contact-primary-school':0.0105, 'email-Enron':0.0537}
if __name__=='__main__':
    st_train, sc_train, indices_train, st_test, sc_test, indices_test, train, test =  get_simplicies_closure(args.data, args.dim)
    st_test, sc_test, B_test = get_boundary_matrices_from_processed_tree(st_test, sc_test, indices_test, 3)

    Ll_test, Lu_test = _get_laplacians(B_test)
    P_B_test, P_L_test, P_U_test = _get_transition_matrix(B_test, Ll_test, Lu_test) 

    g, netxG = _get_graph(sc_test[1], num_nodes= len(sc_test[0]))
    netxG.add_nodes_from(np.setdiff1d(np.arange(1,max(list(netxG.nodes()))), np.array(list(netxG.nodes()))))
    g = g.to(args.device)
    index = {'B':0, 'C':1, 'L':2, 'U':3}
    X = _get_simplex_features(sc_test[1:], g.ndata['features'])

    Psi = scattering_transform(X, P_B_test, P_L_test, P_U_test, index, args.J, args.include_boundary)

    Psi_Psi = []
    for PsiX in Psi:
        Psi_Psi.append(scattering_transform(PsiX, P_B_test, P_L_test, P_U_test, index, args.J, args.include_boundary))

    Phi = []
    for k in range(len(X)):
        Phi_k = X[k]
        for j in range(args.J):
            Phi_k = torch.cat((Phi_k, Psi[j][k]),axis=1)
        for j in range(args.J):
            for _j in range(args.J):
                Phi_k = torch.cat((Phi_k, Psi_Psi[j][_j][k]),axis=1)
        Phi.append(Phi_k)

    X = Phi[0].cpu().detach().numpy()[test[:,:-1]].sum(axis=1)
    y = test[:,-1]
    score = []
    for i in range(10):
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=args.split, stratify=y)
        classifier = SVC(C = 1.2, class_weight={1:len(np.where(y_train)[0])/len(y_train), 0:len(np.where(y_train==0)[0])/len(y_train)}, kernel='rbf', probability=True)
        classifier.fit(X_train, y_train)
        y_score = classifier.predict_proba(X_test)[:, 1]
        average_precision = average_precision_score(y_test, y_score)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        auc_precision_recall = auc(recall, precision)
        score.append(auc_precision_recall/rb[args.data])
    score = np.array(score)
    print(f"Score = {score.mean()}, Standard deviation = {score.std()}")
    