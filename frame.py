import numpy as np
from utils import normalize_points

class Frame(object):

    def __init__(self, slam_map, img, K, extractor):
        self.img = img
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.H = img.shape[0]
        self.W = img.shape[1]        
        
        self.matches = dict()   # dictionary to store matches with other frames
        self.pts3d = dict()     # dictionary to store 3D point corresponding to matches

        self.kps, self.des = extractor.extract(img)
        self.pts_unnorm = np.array(extractor.keypointsToPoints(self.kps))
        self.pts = normalize_points(self.pts_unnorm, self.Kinv)
        self.Rt = np.eye(4)
        self.id = len(slam_map.frames)
        slam_map.frames.append(self)

    def addMatches(self, other_frame, idx1, idx2, pts3d):
        self.matches[other_frame.id] = {'idx1': idx1, 'idx2': idx2}
        self.pts3d[other_frame.id] = pts3d

    def getMatchesWith(self, other_frame):
        if other_frame.id in self.matches:
            ret = self.matches[other_frame.id]['idx1'], \
                    self.matches[other_frame.id]['idx2'], \
                    self.pts3d[other_frame.id]
            return ret
        else:
            return None, None, None