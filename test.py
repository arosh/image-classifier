# coding: utf-8
from __future__ import division, print_function, unicode_literals

import os
import sys
import string
import random

import numpy
import cv2
from sklearn.externals import joblib
import Image
import ImageDraw
import ImageFont

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
    clf = joblib.load('svc.pkl')
    k = 300
    km = joblib.load('kmeans{}.pkl'.format(k))
    scaler = joblib.load('scaler.pkl')
    svc_scaler = joblib.load('svc_scaler.pkl')
    orb = cv2.ORB()
    cap = cv2.VideoCapture(0)

    window_name = 'Image Classifier'
    info_window_name = 'ORB feature'
    dirname = 'additional'
    font = ImageFont.truetype(
            '/usr/share/fonts/truetype/takao-gothic/TakaoExGothic.ttf',
            24)

    while True:
        _, frame = cap.read()

        keycode = cv2.waitKey(1000 // 15)

        if keycode == 1048691:  # 's' key
            random_str = ''.join(random.choice(string.ascii_lowercase) for _ in xrange(6))
            filename = os.path.join(dirname, random_str + '.jpg')
            cv2.imwrite(filename, frame)
            print(filename)

        if keycode == 1048603:  # ESC key
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        if des is None:
            text = '電子部品を置いてください'
        else:
            des = scaler.transform(des.astype(numpy.float))
            bovw = bovw_feature(des, km).astype(numpy.float)
            bovw = svc_scaler.transform(bovw)
            label = clf.predict(bovw)[0]
            text = '予測結果：' + common.targets_ja[label]

        pil_image = Image.fromarray(numpy.uint8(frame))
        pil_draw  = ImageDraw.Draw(pil_image)
        pil_draw.text((20, 20), text, font=font, fill='black')
        # cv2.putText(frame, 'predict: ' + text, (0, 30),
        #             cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), thickness=2)
        frame = numpy.asarray(pil_image)
        cv2.imshow(window_name, frame)

        frame2 = cv2.drawKeypoints(gray, kp)
        cv2.imshow(info_window_name, frame2)

if __name__ == '__main__':
    main()
