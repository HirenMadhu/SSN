import argparse
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import random

import torch

from preprocessing.simplicial_construction import load_variable
from model.scattering import scattering_tp
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SSN')

parser.add_argument('--data', type=str, default='syn', help='Name of dataset.')
parser.add_argument('--J', type=int, default=4, help='Maximum scale')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    print(args)
    datapath = 'data/tp/'+args.data+'/'
    if(args.data == 'mesh'):
        X = load_variable(datapath+'edge_attributes_mesh')

        train_mask =  np.zeros((X.shape[0],))
        n_ = int(0.8*X.shape[0])
        for i in range(n_): train_mask[random.randint(0,(X.shape[0])-1)]=1
        test_mask =  np.zeros((X.shape[0],))
        test_mask[list(np.squeeze(np.where(train_mask==0)))]=1

        X_tr = X[train_mask!=0]
        X_test = X[test_mask!=0]

        last_nodes = load_variable(datapath+'last_nodes_mesh')
        target_nodes = load_variable(datapath+'target_nodes_mesh')

        Y = load_variable(datapath+'targets_mesh')
        y_tr = np.array(Y)[train_mask!=0]
        y_tr = torch.squeeze(torch.Tensor(np.array([[int(y_tr[i][0]-1)] for i in range(len(y_tr))])))
        y_test = np.array(Y)[test_mask!=0]
        y_test = torch.squeeze(torch.Tensor(np.array([[y_test[i][0]-1] for i in range(len(y_test))])))
        B1 = load_variable(datapath+'B1_mesh')
        B2 = load_variable(datapath+'B2_mesh')
        G = load_variable(datapath+'G_mesh')
    elif(args.data == 'ocean'):
        X = np.load(datapath+'flows_in.npy')

        train_mask = np.load(datapath+'train_mask.npy')
        test_mask = np.load(datapath+'test_mask.npy')
        X_tr = X[train_mask!=0]
        X_test = X[test_mask!=0]

        last_nodes = np.load(datapath+'last_nodes.npy')
        target_nodes = np.load(datapath+'target_nodes.npy')
        Y = np.load(datapath+'targets.npy')
        y_tr = [list(np.squeeze(Y[np.where(train_mask!=0)])[i]) for i in range(len(Y[np.where(train_mask!=0)]))]
        y_tr = torch.squeeze(torch.Tensor(np.array(y_tr).argmax(1)))
        y_test = [list(np.squeeze(Y[test_mask!=0])[i]) for i in range(len(Y[test_mask!=0]))]
        y_test = torch.squeeze(torch.Tensor(np.array(y_test).argmax(1)))
        B1 = np.load(datapath+'B1.npy')
        B2 = np.load(datapath+'B2.npy')
        G = load_variable(datapath+'G_undir.pkl')
    elif(args.data == 'syn'):
        X = np.load(datapath+'flows_in.npy')
        train_mask = np.load(datapath+'train_mask.npy')
        test_mask = np.load(datapath+'test_mask.npy')
        X_tr = X[train_mask!=0]
        X_test = X[test_mask!=0]
        last_nodes = np.load(datapath+'last_nodes.npy')
        target_nodes = np.load(datapath+'target_nodes.npy')
        Y = np.load(datapath+'targets.npy')
        y_tr = [list(np.squeeze(Y[np.where(train_mask!=0)])[i]) for i in range(len(Y[np.where(train_mask!=0)]))]
        y_tr = torch.squeeze(torch.Tensor(np.array(y_tr).argmax(1)))
        y_test = [list(np.squeeze(Y[test_mask!=0])[i]) for i in range(len(Y[test_mask!=0]))]
        y_test = torch.squeeze(torch.Tensor(np.array(y_test).argmax(1)))
        B1 = np.load(datapath+'B1.npy')
        B2 = np.load(datapath+'B2.npy')
        G = load_variable(datapath+'G_undir.pkl')
        

    # B1 = np.linalg.pinv(np.diag(B1.sum(1)))@B1
    # B2 = np.linalg.pinv(np.diag(B2.sum(1)))@B2

    N0 = (abs(B1@B1.T).shape)[0]
    N1 = (abs(B2@B2.T).shape)[0]
    N2 = (abs(B2.T@B2).shape)[0]
    print('N0,N1,N2',N0,N1,N2)
    print('X_tr',X_tr.shape)

    x1_0 = np.squeeze(X_tr)
    x1_0_test = np.squeeze(X_test)
    psi_x1_0 = scattering_tp(x1_0, B1, B2, args.J)
    psi_x1_0_test = scattering_tp(x1_0_test, B1, B2, args.J)

    Z_ = []     
    for l in range(len(last_nodes)): #200 for buoy 
        i = last_nodes[l]
        Z__ = np.zeros((B1.shape[0]))
        Z__[[int(j) for j in G.neighbors(i)]]=1
        Z_.append(list(Z__))

    Z_ = np.array(Z_)
    Z_tr_ = Z_[train_mask!=0]
    Z_test = Z_[test_mask!=0]

    X_train = x1_0@B1.T*Z_tr_
    X_test = x1_0_test@B1.T*Z_test
    print(x1_0.shape)   
    print(x1_0_test.shape)
    print(X_train.shape)   
    print(X_test.shape)
    accs = []
    for i in range(10):
        #classifier = SVC(C=3 ,kernel='rbf')
        classifier = LogisticRegression()
        classifier.fit(X_train, y_tr.cpu().detach().numpy())
        accs.append(accuracy_score(y_test.cpu().detach().numpy(), classifier.predict(X_test)))
    accs = np.array(accs)
    print(f"Accuracy = {accs.mean()}, Standard deviation = {accs.std()}")