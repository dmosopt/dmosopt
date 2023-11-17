# -*- coding: utf-8 -*-
"""
Definitions of feasibility models.
"""

import sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


class LogisticFeasibilityModel(object):
    def __init__(self, X, C):
        N = C.shape[1]
        self.clfs = []
        self.X = X
        for i in range(N):
            c_i = (C[:, i] > 0.0).astype(int)
            clf = None
            if len(np.unique(c_i)) > 1:
                pca = PCA()
                scaler = StandardScaler()
                ppl = make_pipeline(
                    pca,
                    scaler,
                    LogisticRegression(tol=0.01, penalty="l1", solver="saga"),
                )
                param_grid = {
                    "pca__n_components": range(1, X.shape[1]),
                    "logisticregression__C": np.logspace(-4, 4, 4),
                }
                clf = GridSearchCV(ppl, param_grid, n_jobs=-1)
                clf.fit(X, c_i)
            self.clfs.append(clf)

    def predict(self, x):
        ps = []
        for clf in self.clfs:
            if clf is not None:
                pred = clf.predict(x)
                ps.append(pred)
            else:
                ps.append(np.ones((x.shape[1],)))

        P = np.column_stack(ps)

        return P

    def predict_proba(self, x):
        probs = []
        for clf in self.clfs:
            if clf is not None:
                prob = clf.predict_proba(x)
                probs.append(prob)
            else:
                probs.append(np.asarray([[0.0, 1.0]] * x.shape[0]))

        Pr = np.stack(probs)

        return Pr

    def rank(self, x):
        pr = self.predict_proba(x)
        mean_pr_feasible = np.mean(pr[:, :, 1], axis=0)
        return mean_pr_feasible
