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


def process_image(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(slam_map, img, K, feature_extractor)
    if frame.id == 0:
        return
    
    idx1, idx2, pts3d = feature_matcher.matchAndUpdate(slam_map.frames[-1], slam_map.frames[-2])

    for p in pts3d:
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