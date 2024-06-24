import numpy as np
from utils import normalize_points

class Frame(object):

    def __init__(self, slam_map, img, K, extractor):
        self.img = img
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.H = img.shape[0]
        self.W = img.shape[1]      
        self.Rt = np.eye(4)  
        
        self.keypoints_unnorm, self.descriptors = extractor.extract(img)
        self.keypoints = normalize_points(self.keypoints_unnorm, self.Kinv)
        self.pts = [None] * len(self.keypoints)
        
        self.id = len(slam_map.frames)
        slam_map.frames.append(self)