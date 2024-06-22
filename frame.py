import numpy as np
from utils import normalize_points

class Frame(object):

    def __init__(self, img, K, extractor):

        self.img = img
        self.K = K
        self.Kinv = np.linalg.inv(K)
        
        self.kps, self.des = extractor.extract(img)
        self.pts = np.array(extractor.keypointsToPoints(self.kps))
        self.pts = normalize_points(self.pts, self.Kinv)
        # self.pts = np.array([])
        self.Rt = np.eye(4)