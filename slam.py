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
points = []

class Point(object):
    
    def __init__(self, location):
        self.frames = []
        self.location = location
        self.idxs = []
        points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def process_image(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(img, K, feature_extractor)
    frames.append(frame)
    if len(frames) < 2:
        return
    
    idx1, idx2, frames[-1].Rt = feature_matcher.match(frames[-1], frames[-2])

    # triangulate points
    frames[-1].Rt = np.dot(frames[-2].Rt, frames[-1].Rt)
    pts4d = triangulate(frames[-1].Rt, frames[-2].Rt, frames[-1].pts[idx1], frames[-2].pts[idx2])

    # reject pts4d wihthout enough parallax
    filter_pts3d_index = np.abs(pts4d[:, 3]) > 0.005
    pts4d = pts4d[filter_pts3d_index]
    pts4d /= pts4d[:, 3].reshape(-1, 1)

    # reject points behind the camera
    pts4d = pts4d[pts4d[:, 2] > 0]
    # print(pts4d)
    for p in pts4d:
        pt = Point(p)
        pt.add_observation(frames[-1], idx1)
        pt.add_observation(frames[-2], idx2)
    
    print(len(points))
    # visualize 3d points using matplotlib
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # aziumth, elevation
    # ax.view_init(-90, -90)
    # # show label x, y, z
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.scatter(pts4d[:, 0], pts4d[:, 1], pts4d[:, 2])
    # plt.show()


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
