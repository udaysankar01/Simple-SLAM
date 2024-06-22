import numpy as np

class Frame(object):

    def __init__(self, img, K, extractor):

        self.img = img
        self.K = K
        self.Kinv = np.linalg.inv(K)
        
        self.kps, self.des = extractor.extract(img)
        self.Rt = None