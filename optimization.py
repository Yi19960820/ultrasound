import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def ista(D, lm1, lm2):
    H1 = la.eye(D.shape[0])
    H2 = la.eye(D.shape[0])
    A = np.hstack((H1, H2))
    Lf = la.norm(A, ord=2)  # largest singular value of A should equal Lf
    L = np.zeros_like(D)
    S = np.zeros_like(D)
    kmax = 20
    for k in range(kmax):
        G1 = (la.eye(D.shape[0])-1/Lf*H1.H@H1)@L - H1.H@H2@S + H1.H@D
        G2 = (la.eye(D.shape[0])-1/Lf*H2.H@H2)@S - H2.H@H1@S + H2.H@D
        L = svt(lm1/Lf, G1)
        S = soft_thresh(lm2/Lf, G2)
    return L, S

def svt(alpha, X):
    # We assume X (L in the algorithm) is low-rank (most singular values are 0) so there may be a faster way to do this
    U, Sigma, Vh = la.svd(X)
    return U@np.max(np.zeros(X.shape[0]), np.diag(Sigma)-alpha)@Vh

def soft_thresh(alpha, X):
    # Compute vector soft-threshholding column-wise (T_a(v)=max(0, 1-a/||v||_2)*v)
    # X (S in the algorithm) is assumed to be sparse, so there may be a faster way to do this
    return np.diag(np.max(np.zeros(X.shape[0]), 1-alpha/la.norm(X, axis=0)))@X