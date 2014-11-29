# coding: utf-8
from __future__ import division, print_function, unicode_literals

import numpy
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def main():
    X = joblib.load('X.pkl').astype(numpy.float)
    y = joblib.load('y.pkl')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # C : 2^-5から2^15くらい
    # gamma : 2^-15から2^3くらい
    param_grid = {
            "kernel":[b"rbf"],
            "C":numpy.logspace(-5.0, 15.0, num=40, base=2.0),
            "gamma":numpy.logspace(-15.0, 3, num=20, base=2.0)}
    clf = GridSearchCV(SVC(), param_grid, n_jobs=-1, cv=5, verbose=1)
    clf.fit(X, y)
    print('score =', clf.best_score_)
    print('params =', repr(clf.best_params_))
    joblib.dump(clf.best_estimator_, 'svc.pkl')
    joblib.dump(scaler, 'svc_scaler.pkl')

    # clf = joblib.load('svc.pkl')
    # print(cross_val_score(clf, X, y, cv=5, n_jobs=-1, verbose=2))

if __name__ == '__main__':
    main()
