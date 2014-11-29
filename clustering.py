# coding: utf-8
from __future__ import division, print_function, unicode_literals
from glob import iglob
from os import path

import numpy
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import common

def main():
    datasets = []
    for target in common.targets:
        print(target, '...', end=' ')
        mat = []
        # numpy行列の拡張子をtxtにするの忘れた…
        for fname in iglob(path.join('features', target, '*.npy')):
            m = numpy.load(fname).astype(numpy.float)
            mat.append(m)
        datasets.append((target, numpy.vstack(mat)))
        print('done ({} images)'.format(len(mat)))

    print('create vstack ...', end=' ')
    X = None
    for i in xrange(len(datasets)):
        _, mat = datasets[i]
        if X is None:
            X = mat
        else:
            X = numpy.vstack((X, mat))
    print('done')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print('create visual words by k-means ...', end=' ')
    k = 300
    km = MiniBatchKMeans(n_clusters=k)
    km.fit(X)
    print('done')

    joblib.dump(km, 'kmeans{}.pkl'.format(k))
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == '__main__':
    main()
