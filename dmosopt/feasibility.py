# -*- coding: utf-8 -*-
"""
Definitions of feasibility models.
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn import svm


class FeasibilityModel(object):

    def __init__(self, X, C):

        N = C.shape[1]
        self.clfs = []
        self.kdt = cKDTree(X)
        
        for i in range(N):
            clf = svm.NuSVC(gamma='auto')
            clfs.append(clf)
            y = (C[:,i] > 0.).astype(int)
            clf.fit(X, y)

        
    def predict(self, x):

        nn_distances, nn = self.kdt.query(x, k=1)
        exp_distances = np.exp(nn_distances[:,0])

        ps = []
        ds = []
        for clf in clfs:
            pred = clf.predict(x)
            cls_distances = np.abs(self.clf[i].decision_function(x))
            delta = cls_distances - exp_distances
            ps.append(pred)
            ds.append(delta)

        P = np.column_stack(ps)
        D = np.column_stack(ds)
        
        return P, D

        
