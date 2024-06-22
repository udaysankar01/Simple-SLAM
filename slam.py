#!/usr/bin/env python
import cv2 
import numpy as np
from feature import FeatureExtractor, FeatureMatcher
from frame import Frame
import g2o
from utils import *

W = 1920//2
H = 1080//2
F = 280
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

feature_extractor = FeatureExtractor(method='shitomasi')
feature_matcher = FeatureMatcher(K, W, H)

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []

    def display(self):
        for frame in self.frames:
            print(frame.id)
            print(frame.Rt)
            print('\n')
        
        # for point in self.points:
        #     print(point.id)


slam_map = Map()

class Point(object):
    
    def __init__(self, slam_map, position):
        self.frames = []
        self.position = position
        self.idxs = []
        
        self.id = len(slam_map.points)
        slam_map.points.append(self)


    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def process_image(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(slam_map, img, K, feature_extractor)
    if frame.id == 0:
        return
    
    idx1, idx2, slam_map.frames[-1].Rt = feature_matcher.match(slam_map.frames[-1], slam_map.frames[-2])

    # triangulate points
    slam_map.frames[-1].Rt = np.dot(slam_map.frames[-2].Rt, slam_map.frames[-1].Rt)
    pts4d = triangulate(slam_map.frames[-1].Rt, slam_map.frames[-2].Rt, slam_map.frames[-1].pts[idx1], slam_map.frames[-2].pts[idx2])

    # reject unwanted points
    filter_pts3d_index = np.abs(pts4d[:, 3]) > 0.005
    pts4d = pts4d[filter_pts3d_index]
    pts4d /= pts4d[:, 3].reshape(-1, 1)
    pts4d = pts4d[pts4d[:, 2] > 0]

    for p in pts4d:
        pt = Point(slam_map, p)
        pt.add_observation(slam_map.frames[-1], idx1)
        pt.add_observation(slam_map.frames[-2], idx2)
    
    slam_map.display()


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
