# coding: utf-8
from __future__ import division, print_function, unicode_literals

import numpy
import cv2
from glob import glob
import os
import shutil

import common

def main():
    src_base  = 'category'
    dest_base = 'features'
    if os.path.isdir(dest_base):
        shutil.rmtree(dest_base)
    if not os.path.isdir(dest_base):
        os.mkdir(dest_base)

    for target in common.targets:
        dirname = os.path.join(dest_base, target)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        for fname in glob(os.path.join(src_base, target, "*.jpg")):
            # グレースケールで読み込む
            gray = cv2.imread(fname, 0)
            orb = cv2.ORB()
            kp, des = orb.detectAndCompute(gray, None)

            root, _ = os.path.splitext(fname)
            destfile = os.path.join(dest_base, target, os.path.basename(root) + '.npy')
            numpy.save(destfile, des)
            print(destfile)


if __name__ == '__main__':
    main()
