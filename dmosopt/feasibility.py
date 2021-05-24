# -*- coding: utf-8 -*-
"""
Definitions of feasibility models.
"""

import sys
import numpy as np
from scipy.spatial import cKDTree
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class FeasibilityModel(object):

    def __init__(self, X, C):

        N = C.shape[1]
        self.clfs = []
        self.kdt = cKDTree(X)
        
        for i in range(N):
            clf = make_pipeline(StandardScaler(), svm.NuSVC(gamma='auto', class_weight='balanced', nu=0.01))
            self.clfs.append(clf)
            y = (C[:,i] > 0.).astype(int)
            clf.fit(X, y)

        
    def predict(self, x):

        nn_distances, nn = self.kdt.query(x, k=1)
        exp_distances = np.exp(nn_distances)

        ps = []
        ds = []
        for clf in self.clfs:
            pred = clf.predict(x)
            cls_distances = np.abs(clf.decision_function(x))
            delta = cls_distances - exp_distances
            ps.append(pred)
            ds.append(delta)

        P = np.column_stack(ps)
        D = np.column_stack(ds)
        
        return P, D

        
