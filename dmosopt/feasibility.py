# -*- coding: utf-8 -*-
"""
Definitions of feasibility models.
"""

import sys
import numpy as np
from scipy.spatial import cKDTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

class FeasibilityModel(object):

    def __init__(self, X, C):

        N = C.shape[1]
        self.clfs = []
        self.kdt = cKDTree(X)
        self.X = X
        for i in range(N):
            clf = make_pipeline(StandardScaler(), SGDClassifier())
            self.clfs.append(clf)
            y = (C[:,i] > 0.).astype(int)
            clf.fit(X, y)

    def sample(self, size):
        
        K = np.cov(self.X.T)
        n = self.X.shape[0]
        sample_size = 1
        if size > n:
            sample_size = int(size / n)
        zlst = []
        count = 0
        for i in range(n):
            z = np.random.multivariate_normal(mean=self.X[i,:], cov=K, size=sample_size)
            if count + sample_size < size:
                zlst.append(z)
                count += sample_size
            else:
                zlst.append(z[:size - count, :])
                count = size

        return np.vstack(zlst)
        
        
        
    def predict(self, x):

        nn_distances, nn = self.kdt.query(x, k=1)
        ED = np.exp(nn_distances)

        ps = []
        ds = []
        for clf in self.clfs:
            pred = clf.predict(x)
            cls_distances = np.abs(clf.decision_function(x))
            ps.append(pred)
            ds.append(cls_distances)

        P = np.column_stack(ps)
        CD = np.column_stack(ds)
        
        return P, CD, ED

        
