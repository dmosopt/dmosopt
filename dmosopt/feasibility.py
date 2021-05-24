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

    def __init__(self, X, C, nu=0.05):

        N = C.shape[1]
        self.clfs = []
        self.kdt = cKDTree(X)
        
        for i in range(N):
            clf = make_pipeline(StandardScaler(), svm.NuSVC(gamma='auto', class_weight='balanced', nu=nu))
            self.clfs.append(clf)
            y = (C[:,i] > 0.).astype(int)
            clf.fit(X, y)

        
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

        
