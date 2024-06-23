import numpy as np
from utils import normalize_points

class Frame(object):

    def __init__(self, slam_map, img, K, extractor):

        self.img = img
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.H = img.shape[0]
        self.W = img.shape[1]
        
        self.kps, self.des = extractor.extract(img)
        self.pts_unnorm = np.array(extractor.keypointsToPoints(self.kps))
        self.pts = normalize_points(self.pts_unnorm, self.Kinv)
        self.Rt = np.eye(4)
        self.id = len(slam_map.frames)
        slam_map.frames.append(self)