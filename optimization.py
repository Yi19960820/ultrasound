import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed()

def ista(D, lm1, lm2, AL, AS):
    H1 = np.eye(D.shape[0])
    H2 = np.eye(D.shape[0])
    A = np.hstack((H1, H2))
    Lf = la.norm(A, ord=2)  # largest singular value of A should equal Lf
    # Lf = 2
    L = np.zeros_like(D, dtype='complex128')
    S = np.zeros_like(D, dtype='complex128')
    kmax = 30
    err = []

    for k in range(kmax):
        G1 = (np.eye(D.shape[0])-1/Lf*dagger(H1)@H1)@L - dagger(H1)@H2@S + dagger(H1)@D
        G2 = (np.eye(D.shape[0])-1/Lf*dagger(H2)@H2)@S - dagger(H2)@H1@S + dagger(H2)@D
        L = svt(lm1/Lf, G1)
        S = soft_thresh(lm2/Lf, G2)
        cost = la.norm(((AL+AS)-(L+S)), 'fro')**2
        err.append(cost/2 + lm1*np.trace(np.sqrt((L.T)@L)) + lm2*la.norm(S, 1))
    return L, S, err

def svt(alpha, X):
    # We assume X (L in the algorithm) is low-rank (most singular values are 0) so there may be a faster way to do this
    U, Sigma, Vh = la.svd(X, full_matrices=False)
    A = U@np.diag(np.maximum(np.zeros(X.shape[0]), Sigma-alpha))
    return A@Vh

def soft_thresh(alpha, X):
    # Compute vector soft-threshholding column-wise (T_a(v)=max(0, 1-a/||v||_2)*v)
    # X (S in the algorithm) is assumed to be sparse, so there may be a faster way to do this
    thresh = 1-alpha/la.norm(X, axis=0)
    A = np.maximum(np.zeros(thresh.shape), thresh)
    return X@np.diag(A).T

def dagger(A):
    return np.conjugate(A).T

if __name__=='__main__':
    rows = 100
    cols = 200
    rank = 3
    nonzero = 200
    sigma = np.random.random(rank)+1e-10
    AL = np.zeros((rows, cols), dtype='complex128')
    for i in range(rank):
        u = np.random.randn(rows)+1j*np.random.randn(rows)
        u = u/la.norm(u)
        v = np.random.randn(cols)
        v = v/la.norm(v)
        AL = AL+sigma[i]*np.outer(u, v)
    AL = AL/la.norm(AL, 'fro')

    AS = np.zeros((rows, cols))
    indices = zip(np.random.randint(rows, size=nonzero), np.random.randint(cols, size=nonzero))
    for (i, j) in indices:
        AS[i, j] = np.random.randn(1)[0]
    
    AS = AS/la.norm(AS, 'fro')*(10**(-0.5))

    X = AL + AS
    AN = np.random.randn(rows*cols).reshape((rows, cols))+1j*np.random.randn(rows*cols).reshape((rows, cols))
    AN = AN/la.norm(AN, 'fro')*la.norm(X, 'fro')*1

    D = X + AN
    L, S, err = ista(D, 1e-4, 2e-3, AL, AS)
    plt.plot(range(1, len(err)+1), err)
    plt.show()
