import torch
import numpy as np
device = torch.device("cuda:0")
def message_passing(X, B, Lu, Ll, include_boundary):
    neighbors = []
    aggregate = []
    for k in range(len(Lu)):
        X_l = torch.zeros(X[k].shape).to(device)
        X_u = torch.zeros(X[k].shape).to(device)
        X_b = torch.zeros(X[k].shape).to(device)
        X_c = torch.zeros(X[k].shape).to(device)
        if(Lu[k] is not None):
            X_u = Lu[k]@X[k]
        if(Ll[k] is not None):
            X_l = Ll[k]@X[k]
        if include_boundary:
            if(B[k] is not None):
                if(k<len(Lu)-1):
                    X_b = B[k]@X[k+1]
            if(k<len(Lu)-1):
                if(B[k+1] is not None):
                    if(k>0):
                        X_c = B[k-1].T@X[k-1]
        neighbors.append([X_b, X_c, X_l, X_u])
        aggregate.append(X[k] + X_b + X_c + X_l + X_u)
    return neighbors, aggregate

def calculate_Z(X, P_B, P_L, P_U, J, include_boundary):
    Z_agg = []
    Z_neigh = []
    for i in range(J):
        if(i==0):
            neigh, agg = message_passing(X, P_B, P_L, P_U, include_boundary)
            Z_agg.append(agg)
            Z_neigh.append(neigh)
        else:
            neigh, agg = message_passing(Z_agg[-1], P_B, P_L, P_U, include_boundary)
            Z_agg.append(agg)
            Z_neigh.append(neigh)
    return Z_agg, Z_neigh

def scattering(X, Z_neigh, index, J):
    psi = []
    for i in index:
        p = []
        for j in range(J+1):
            out = []
            if(j==0):
                for k in range(len(X)):
                    out.append(torch.abs(X[k] - Z_neigh[j][k][index[i]]))
            if(j==J):
                for k in range(len(X)):
                    out.append(torch.abs(Z_neigh[-1][k][index[i]]))
            else:
                for k in range(len(X)):
                    out.append(torch.abs(Z_neigh[j-1][k][index[i]] - Z_neigh[j][k][index[i]]))
            p.append(out)
        psi.append(p)
    return psi

def agg(psi, X, index, J):
    Psi = []
    for j in range(J):
        psi_j = []
        for k in range(len(X)):
            psi_j.append((X[k]+psi[index['B']][j][k]+psi[index['C']][j][k]+psi[index['L']][j][k]+psi[index['U']][j][k])/5)
        Psi.append(psi_j)
    return Psi

def scattering_transform(X, P_B, P_L, P_U, index, J, include_boundary = True):
    Z_agg, Z_neigh = calculate_Z(X, P_B, P_L, P_U, J, include_boundary)
    psi = scattering(X, Z_neigh, index, J)
    Psi_j = agg(psi, X, index, J)
    return Psi_j

def scattering_tp(x1_0, B1, B2, J):
    x1 = []
    x1.append(x1_0)
    for i in range(J):
        x1_tr = x1[-1]@(B2@B2.T) + x1[-1]@(B1.T@B1)
        x1.append(x1_tr)
    psi = []
    for i in range(J):            
        if(i==J-1):
            psi.append(np.abs(x1[-1]))
        else:
            psi.append(np.abs(x1[i-1]-x1[i]))
    return psi

def scattering_transform_tp(x1_0, B1, B2, J):
    psi = scattering(x1_0, B1, B2, J)
    psi_psi = [scattering(psi[i], B1, B2, J) for i in range(len(psi))]
    phi = [x1_0 for i in range(J)]
    for i in range(J):
        phi[i]+=psi[i]
    for i in range(J):
        for j in range(J):
            phi[j]+=psi_psi[i][j]
    return phi