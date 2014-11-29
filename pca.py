# coding: utf-8
from __future__ import division, print_function, unicode_literals

import numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.externals import joblib
import common
import matplotlib.pyplot as plt 

def main():
    X = joblib.load('X.pkl').astype(numpy.float)
    y = joblib.load('y.pkl')
    X = scale(X)
    pca = PCA(n_components=2)
    pca.fit(X)
    colors = 'bgrcmykwbg'
    for i in [0, 1]:
        x = pca.transform(X[y == i, :])
        c = colors[i]
        plt.scatter(x[:,0], x[:,1], c=c, label=common.targets[i])
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
