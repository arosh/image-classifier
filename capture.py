# coding: utf-8
from __future__ import division, print_function, unicode_literals

import argparse
import os
import os.path
import random
import string
import sys

import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('category', type=str)
    args = parser.parse_args()

    category = args.category
    window_name = sys.argv[0]

    dirname = os.path.join('category', args.category)

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        cv2.imshow(window_name, frame)
        keycode = cv2.waitKey(1000 // 30)
        if keycode == -1:
            continue
        # print('keycode = ', repr(keycode))
        if keycode == 1048603:  # ESC key
            break
        if keycode == 1048608: # Space key
            random_str = ''.join(random.choice(string.ascii_lowercase) for _ in xrange(6))
            while os.path.isfile(os.path.join(dirname, random_str + '.jpg')):
                random_str = ''.join(random.choice(string.ascii_lowercase) for _ in xrange(6))
            filename = os.path.join(dirname, random_str + '.jpg')
            cv2.imwrite(filename, frame)
            print(filename)

if __name__ == '__main__':
    main()
