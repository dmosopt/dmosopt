# -*- coding: utf-8 -*-
"""
Definition of Sampling Methods
"""

import numpy as np
from dmosopt import GLP
from scipy.stats import qmc

def SobolDesign(n,s,local_random):
    ''' Generate Sobol Design
        n: number of samples
        s: number of dimensions
    '''

    sampler = qmc.Sobol(d=s,scramble=True,seed=local_random)
    m=10 # start at 1024 samples
    while pow(2,m) < n:
        m = m+1
    sample = sampler.random_base2(m)
    return sample[:n]

def MonteCarloDesign(n,s,local_random):
    ''' Generate Monte Carlo Design
        n: number of samples
        s: number of dimensions
    '''
    return local_random.random(size=(n,s))

def LatinHypercubeDesign(n,s,local_random):
    ''' Generate Latin Hypercube Design
        n: number of samples
        s: number of dimensions
    '''
    gen = qmc.LatinHypercube(d=s, seed=local_random)
    sample = gen.random(n=n)
    return sample


def SymmetricLatinHypercubeDesign(n,s,local_random):
    ''' Generate Symmetric Latin Hypercube Design
        n: number of samples
        s: number of dimensions
    '''
    delta = (1.0 / n) * np.ones(s)
    x = np.ndarray([n,s])

    for j in range(s):
        for i in range(n):
            x[i,j] = ((2.0 * (i + 1) - 1) / 2.0) * delta[j]
    p = np.zeros([n, s], dtype = int);

    p[:, 0] = np.arange(n)
    if n % 2 == 0:
        k = int(n / 2)
    else:
        k = int((n - 1) / 2)
        p[k, :] = (k + 1) * np.ones((1, s))

    for j in range(1, s):
        p[0:k, j] = local_random.permutation(np.arange(k))
        for i in range(int(k)):
            if local_random.random() < 0.5:
                p[n - 1 - i, j] = n - 1 - p[i, j]
            else:
                p[n - 1 - i, j] = p[i, j]
                p[i, j] = n - 1 - p[i, j]

    res = np.zeros([n, s])
    for j in range(s):
        for i in range(n):
            res[i, j] = x[p[i, j], j]

    return res

def rmtrend(x,y):
    '''remove the trend between x and y from y'''
    xm = x - x.mean()
    ym = y - y.mean()
    b  = (xm * ym).sum()/(xm ** 2.).sum()  # b = (X'X)^(-1)X'y
    z  = y - b * xm
    return z

def rand2rank(r):
    '''transfer random number in [0,1] to integer number '''
    n = len(r)
    x = np.ndarray(n)
    x[r.argsort()] = np.array(range(n))
    return x

def decorr(x,n,s):
    '''Ranked Gram-Schmidt (RGS) de-correlation iteration'''
    # Forward ranked Gram-Schmidt step:
    for j in range(1,s):
        for k in range(j):
            z = rmtrend(x[:,j],x[:,k])
            x[:,k] = (rand2rank(z) + 0.5) / n
    # Backward ranked Gram-Schmidt step:
    for j in range(s-2,-1,-1):
        for k in range(s-1,j,-1):
            z = rmtrend(x[:,j],x[:,k])
            x[:,k] = (rand2rank(z) + 0.5) / n
    return x



def LatinHypercubeDesignDecorrelation(n,s,local_random,maxiter = 5):
    ''' Generate Latin Hypercube Design with de-correlation
        n: number of samples
        s: number of dimensions
        maxiter: number of iterations of de-correlation
    '''
    x = LatinHypercubeDesign(n,s,local_random)
    for i in range(maxiter):
        x = decorr(x,n,s)
    return x

def SymmetricLatinHypercubeDesignDecorrelation(n,s,local_random,maxiter = 5):
    ''' Generate Symmetric Latin Hypercube Design with de-correlation
        n: number of samples
        s: number of dimensions
        maxiter: number of iterations of de-correlation
    '''
    x = SymmetricLatinHypercubeDesign(n,s,local_random)
    for i in range(maxiter):
        x = decorr(x,n,s)
    return x

def GoodLatticePointsDesign(n,s,local_random):
    ''' Generate Good Lattice Points Design
        n: number of samples
        s: number of dimensions
    '''
    return GLP.sample(n,s,local_random)

def GoodLatticePointsDesignDecorrelation(n,s,local_random,maxiter = 5):
    ''' Generate Good Lattice Points Design with de-correlation
        n: number of samples
        s: number of dimensions
        maxiter: number of iterations of de-correlation
    '''
    x = GLP.sample(n,s,local_random)
    for i in range(maxiter):
        x = decorr(x,n,s)
    return x

def mc(n,s,local_random):
    ''' short name of MonteCarloDesign'''
    return MonteCarloDesign(n,s,local_random)

def lh(n,s,local_random,maxiter = 0):
    ''' short name of LatinHypercubeDesign'''
    if maxiter == 0:
        return LatinHypercubeDesign(n,s,local_random)
    else:
        return LatinHypercubeDesignDecorrelation(n,s,local_random,maxiter)

def slh(n,s,local_random,maxiter = 0):
    ''' short name of SymmetricLatinHypercubeDesign'''
    if maxiter == 0:
        return SymmetricLatinHypercubeDesign(n,s,local_random)
    else:
        return SymmetricLatinHypercubeDesignDecorrelation(n,s,local_random,maxiter)

def glp(n,s,local_random,maxiter = 0):
    ''' short name of GoodLatticePointsDesign'''
    if maxiter == 0:
        return GoodLatticePointsDesign(n,s,local_random)
    else:
        return GoodLatticePointsDesignDecorrelation(n,s,local_random,maxiter)

def sobol(n,s,local_random):
    ''' short name of SobolDesign'''
    return SobolDesign(n,s,local_random)
    
