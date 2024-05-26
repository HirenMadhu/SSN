import numpy as np
from tqdm import tqdm
import copy
import gudhi
import torch
import pickle
device = torch.device("cuda:0")

def process_simplex_tree(simplex_tree, num_nodes):
    sc = [list() for _ in range(simplex_tree.dimension()+1)]
    sc[0] = [[i] for i in range(num_nodes)]
    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        sc[len(simplex)-1].append(np.array(simplex))
    indices = []
    for i in range(len(sc)):
        sc[i] = np.array(sc[i])
        sc[i] = np.unique(sc[i],axis=0)
        index = {}
        for k,j in enumerate(sc[i]):
            index[frozenset(j)] = k
        indices.append(index)
    sc[0] = np.unique(np.array(sc[0]),axis=0)
    return simplex_tree, sc, indices

def get_simplicies_closure(data,dim):
    with open('data/closure/'+ data + '/' + data +'-nverts.txt', 'r') as f:
        nverts = np.array(list(map(lambda x:int(x), f.readlines())))
    with open('data/closure/'+ data + '/' + data +'-simplices.txt', 'r') as f:
        simplices = np.array(list(map(lambda x:int(x), f.readlines())))
    print(simplices.min())
    train = np.load('data/closure/'+ data + '/' + data +'-train.npy', allow_pickle=False)
    test = np.load('data/closure/'+ data + '/' + data +'-test.npy', allow_pickle=False)
    bipartite_graph = np.zeros((len(nverts), np.max(simplices))).astype('bool')
    seen = 0
    for i in range(len(nverts)):
        nodes = []
        for j in range(seen, seen+nverts[i]):
            nodes.append(simplices[j]-1)
        bipartite_graph[i,nodes]=1
        seen+=nverts[i]

    simplex_tree_train = gudhi.SimplexTree()
    for i in bipartite_graph[:int(len(bipartite_graph)*0.8)]:
        simplex = np.where(i)[0]
        if(len(simplex)<=dim+1):
            simplex_tree_train.insert(np.array(simplex))
    simplex_tree_train.prune_above_dimension(dim)
    st_train, sc_train, indices_train = process_simplex_tree(simplex_tree_train, simplices.max())

    simplex_tree_test = gudhi.SimplexTree()
    for i in bipartite_graph:
        simplex = np.where(i)[0]
        if(len(simplex)<=dim+1):
            simplex_tree_test.insert(simplex)
    simplex_tree_test.prune_above_dimension(dim)
    st_test, sc_test, indices_test = process_simplex_tree(simplex_tree_test, simplices.max())


    return st_train, sc_train, indices_train, st_test, sc_test, indices_test, train, test

def get_simplicies(data,dim):
    with open('data/node classification/node-labels-'+ data +'.txt', 'r') as f:
        labels = np.array(list(map(lambda x:int(x), f.readlines())))
    print("Read the labels")
    simplicies = []
    num_nodes = 0
    print("Reading the simplicies")
    with open('data/node classification/hyperedges-'+ data +'.txt', 'r') as f:
        for i in f.readlines():
            simplicies.append(np.array([int(y) for y in i[:-1].split(',')]))
            if(simplicies[-1].max()>num_nodes):
                num_nodes = simplicies[-1].max()
    simplex_tree = gudhi.SimplexTree()
    print("Creating tree")
    for i in simplicies:
        if len(i)<=dim+1:
            simplex_tree.insert(np.array(i)-1)
    simplex_tree.prune_above_dimension(dim)
    simplex_tree, sc, indices = process_simplex_tree(simplex_tree, num_nodes)
    return simplex_tree, sc, indices, labels

def get_simplicies_sp(data,dim):
    with open('data/simplex prediction/cat-edge-'+data+'/hyperedge-labels.txt', 'r') as f:
        labels = np.array(list(map(lambda x:int(x), f.readlines())))
    print("Read the labels")
    simplicies = []
    num_nodes = 0
    print("Reading the simplicies")
    with open('data/simplex prediction/cat-edge-'+data+'/hyperedges.txt', 'r') as f:
        for i in f.readlines():
            simplicies.append(np.array([int(y) for y in i[:-1].split(' ')]))
            if(simplicies[-1].max()>num_nodes):
                num_nodes = simplicies[-1].max()
    simplex_tree = gudhi.SimplexTree()
    print("Creating tree")
    for i in tqdm(simplicies):
        if len(i)<=dim+1:
            simplex_tree.insert(np.array(i)-1)
    simplex_tree.prune_above_dimension(dim)
    simplex_tree, sc, indices = process_simplex_tree(simplex_tree, num_nodes)
    return simplex_tree, sc, indices, labels, simplicies
def get_boundary_matrices_sp(data, dim):
  simplex_tree, sc, indices, labels, simplicies = get_simplicies_sp(data, dim)
  boundry_matrices = []
  for i in range(1,dim+1):
      print(f"Computing boundary matrix for dimension {i}")
      boundry_matrix = np.zeros((len(sc[i-1]),len(sc[i])), dtype=bool)
      for m,j in enumerate(sc[i]):
        idx = np.arange(1, i+1) - np.tri(i+1, i, k=-1, dtype=bool)
        for k in idx:
            boundry_matrix[indices[i-1][frozenset(j[k])], indices[i][frozenset(j)]] = 1
              #boundry_matrix[np.where(np.all(simplicies[i-1] == j[k], axis=1)), np.where(np.all(simplicies[i] == j, axis=1))] = 1
      boundry_matrices.append(boundry_matrix)
  return simplex_tree, sc[:dim], boundry_matrices, labels, simplicies

def get_boundary_matrices(data, dim):
  simplex_tree, sc, indices, labels = get_simplicies(data, dim)
  boundry_matrices = []
  for i in range(1,dim+1):
      print(f"Computing boundary matrix for dimension {i}")
      boundry_matrix = np.zeros((len(sc[i-1]),len(sc[i])), dtype=bool)
      for m,j in enumerate(sc[i]):
        idx = np.arange(1, i+1) - np.tri(i+1, i, k=-1, dtype=bool)
        for k in idx:
            boundry_matrix[indices[i-1][frozenset(j[k])], indices[i][frozenset(j)]] = 1
      boundry_matrices.append(boundry_matrix)
  return simplex_tree, sc[:dim], boundry_matrices, labels

def get_boundary_matrices_from_processed_tree(simplex_tree, sc, indices, dim):
  boundry_matrices = []
  for i in range(1,dim+1):
      boundry_matrix = np.zeros((len(sc[i-1]),len(sc[i])), dtype=bool)
      for m,j in enumerate(sc[i]):
        idx = np.arange(1, i+1) - np.tri(i+1, i, k=-1, dtype=bool)
        for k in idx:
            boundry_matrix[indices[i-1][frozenset(j[k])], indices[i][frozenset(j)]] = 1
      boundry_matrices.append(boundry_matrix)
  return simplex_tree, sc[:dim], boundry_matrices

def _get_simplex_features(simplicies,features):
    X = []
    X.append(features)
    for i in simplicies:
        X.append(features[i].sum(axis=1).clip(0,1))
    return X

def _get_laplacians(boundary_matrices):
  for i,k in enumerate(boundary_matrices):
    boundary_matrices[i] = torch.FloatTensor(k).to(device)
  lower_laplacians = [None]*len(boundary_matrices)
  upper_laplacians = [None]*len(boundary_matrices)
  for i in range(1,len(boundary_matrices)):
    lower_laplacians[i] = boundary_matrices[i-1].T@boundary_matrices[i-1]
  for i in range(0,len(boundary_matrices)-1):
    upper_laplacians[i] = boundary_matrices[i]@boundary_matrices[i].T
  return lower_laplacians, upper_laplacians

def _get_transition_matrix(boundry_matrices, lower_laplacians, upper_laplacians):
    P_B = []
    P_U = []
    P_L = []
    for i in range(len(boundry_matrices)):
        P_B.append((torch.linalg.pinv(torch.diag(boundry_matrices[i].sum(axis=1)))@boundry_matrices[i]).to(device))
    for i in range(len(upper_laplacians)):
        if(upper_laplacians[i] is not None):
            P_U.append((upper_laplacians[i]@torch.linalg.pinv(torch.diag(upper_laplacians[i].sum(axis=1)))).to(device))
        else:
            P_U.append(None)
    for i in range(len(lower_laplacians)):
        if(lower_laplacians[i] is not None):
            P_L.append((lower_laplacians[i]@torch.linalg.pinv(torch.diag(lower_laplacians[i].sum(axis=1)))).to(device))
        else:
            P_L.append(None)
    return P_B, P_L, P_U

def load_variable(filename):
  return pickle.load(open(filename,'rb')) 