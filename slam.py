#!/usr/bin/env python
import cv2 
import numpy as np
from feature import FeatureExtractor, FeatureMatcher
from frame import Frame
import g2o
from utils import *

import matplotlib.pyplot as plt
W = 1920//2
H = 1080//2
F = 280
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

feature_extractor = FeatureExtractor(method='shitomasi')
feature_matcher = FeatureMatcher(K, W, H)
frames = []

def process_image(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(img, K, feature_extractor)
    if len(frames) == 0:
        frame.Rt = np.eye(4)
        frames.append(frame)
        return
    frames.append(frame)
    matches, frames[-1].Rt = feature_matcher.match(frames[-1], frames[-2])

    # triangulate points
    frames[-1].Rt = np.dot(frames[-2].Rt, frames[-1].Rt)
    pts4d = cv2.triangulatePoints(frames[-1].Rt[:3], frames[-2].Rt[:3], matches[:, 0].T, matches[:, 1].T).T

    # reject pts4d wihthout enough parallax
    filter_pts3d_index = np.abs(pts4d[:, 3]) > 0.005
    pts4d = pts4d[filter_pts3d_index]
    pts4d /= pts4d[:, 3].reshape(-1, 1)
    print(pts4d.shape)

    # reject points behind the camera
    pts4d = pts4d[pts4d[:, 2] > 0]
    print(pts4d.shape)
    
    # visualize 3d points using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # aziumth, elevation
    ax.view_init(-90, -90)
    # show label x, y, z
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(pts4d[:, 0], pts4d[:, 1], pts4d[:, 2])
    plt.show()


def main():
    cap = cv2.VideoCapture('videos/test.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            process_image(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
