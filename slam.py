#!/usr/bin/env python
import cv2 
import numpy as np
import g2o
import os
import sys
import argparse

from utils import *
from mappoint import Map, Point
from frame import Frame
from display import Display 
from feature import FeatureExtractor, FeatureMatcher

# hard set this!!!!
# F = 280
F = int(os.getenv('F', 800))

# camera intrinsic properties
W = 1920//2
H = 1080//2
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])


feature_extractor = FeatureExtractor(method='shitomasi')
feature_matcher = FeatureMatcher(K, W, H)
slam_map = Map()   
slam_map.create_viewer() if os.getenv('D3D') is not None else None 
display = Display(W, H) if os.getenv('D2D') is not None else None


def triangulate(pose1, pose2, pts1, pts2):
    points_4d = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)                 # not sure why we need to invert the pose
    pose2 = np.linalg.inv(pose2)
    for i in range(pts1.shape[0]):
        A = np.zeros((4, 4))
        A[0] = pts1[i, 0] * pose1[2] - pose1[0]
        A[1] = pts1[i, 1] * pose1[2] - pose1[1]
        A[2] = pts2[i, 0] * pose2[2] - pose2[0]
        A[3] = pts2[i, 1] * pose2[2] - pose2[1]

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        points_4d[i] = X

    return points_4d


def process_image(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(slam_map, img, K, feature_extractor)
    if frame.id == 0:
        return
    
    idx1, idx2, slam_map.frames[-1].Rt = feature_matcher.match(slam_map.frames[-1], slam_map.frames[-2])
    # print(slam_map.frames[-1].Rt)

    # triangulate points
    slam_map.frames[-1].Rt = np.dot(slam_map.frames[-2].Rt, slam_map.frames[-1].Rt)
    # pts4d = triangulate(slam_map.frames[-1].Rt, slam_map.frames[-2].Rt, slam_map.frames[-1].pts[idx1], slam_map.frames[-2].pts[idx2])
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
    
    # 2D visualization
    if display is not None:
        display.showKeypointsAndMatches(slam_map, idx1, idx2)

    # 3D visualization
    slam_map.display()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--video', '-v', required=True, help='Path to video file')
    args = args.parse_args()
    print(args.video)

    cap = cv2.VideoCapture(args.video)
    
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