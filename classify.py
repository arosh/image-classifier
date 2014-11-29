# coding: utf-8
from __future__ import division, print_function, unicode_literals

import argparse
from glob import iglob
from os import path

import cv2
import numpy
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import common

def bovw_feature(image_features, km):
    '''画像の特徴量(SIFTやSURF, ORBなど)を入力とし、
    BoVWに変換する
    '''
    labels = km.predict(image_features)
    histogram = numpy.bincount(labels, minlength=km.n_clusters)
    # なぜか知らないけど正規化しないほうが性能が良い
    return histogram # / numpy.sum(histogram)

def main():
    k = 300
    km = joblib.load('kmeans{}.pkl'.format(k))
    scaler = joblib.load('scaler.pkl')

    X = None
    y = None
    for i in xrange(len(common.targets)):
        target = common.targets[i]
        print(target, '...', end=' ')
        Xs = []
        ys = []
        # numpy行列の拡張子をtxtにするの忘れた
        for fname in iglob(path.join('features', target, '*.npy')):
            m = numpy.load(fname).astype(numpy.float)
            m = scaler.transform(m)
            bovw = bovw_feature(m, km)
            Xs.append(bovw)
            ys.append(i)

        if X is None:
            X = numpy.vstack(Xs)
            y = numpy.hstack(ys)
        else:
            X = numpy.vstack((X, numpy.vstack(Xs)))
            y = numpy.hstack((y, numpy.hstack(ys)))

        print('done ({} images)'.format(len(ys)))

    joblib.dump(X, 'X.pkl')
    joblib.dump(y, 'y.pkl')

if __name__ == '__main__':
    main()
