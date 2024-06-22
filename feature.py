import cv2
import math
import numpy as np
from display import Display
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
from utils import *
np.set_printoptions(suppress=True)


class FeatureExtractor(object):

    def __init__(self, method='shitomasi'):
        self.supported_methods = ['orb', 'shitomasi']
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported method: {method}. Choose method from {self.supported_methods} ")
        self.method = method
        self.grid_size = (4, 4)
        self.orb = cv2.ORB_create(500)
    
    def extract(self, img):
        if self.method == 'orb':
            kps, des = self.extractOrb(img)
        elif self.method == 'shitomasi':
            kps, des = self.extractShiTomasi(img)
        
        return kps, des
    
    def keypointsToPoints(self, keypoints):
        points = []
        for keypoint in keypoints:
            points.append(keypoint.pt)
        points = np.array(points)
        return points
    
    def extractShiTomasi(self, img):
        """
        Extracts Shi Tomasi features from the image.

        Parameters
        ----------
        img : np.array
            The image to extract features from.

        Returns
        -------
        keypoints : list
            List of keypoints.
        descriptors : np.array
            The descriptors of the keypoints.
        """
        features = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 3000, 0.01, 3)
        keypoints = [cv2.KeyPoint(x=feature[0][0], y=feature[0][1], size=20) for feature in features]
        keypoints, descriptors = self.orb.compute(img, keypoints)
        keypoints = np.array(keypoints)
        descriptors = np.array(descriptors)
        return keypoints, descriptors

    def extractOrb(self, img):
        """
        Extracts ORB features from the image using grid based technique.
        Also assumes a grid size of 4x4.

        Parameters
        ----------
        img : np.array
            The image to extract features from.

        Returns
        -------
        keypoints : list
            List of keypoints.
        descriptors : np.array
            The descriptors of the keypoints.
        """
        grid_size = (4, 4)
        keypoints = []
        descriptors = None
        grid_h, grid_w = img.shape[0] // grid_size[0], img.shape[1] // grid_size[1]

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x_start, x_end = j * grid_w, (j + 1) * grid_w
                y_start, y_end = i * grid_h, (i + 1) * grid_h

                grid_img = img[y_start:y_end, x_start:x_end]

                kp, des = self.orb.detectAndCompute(grid_img, None)

                if kp:
                    for k in kp:
                        k.pt = (k.pt[0] + x_start, k.pt[1] + y_start)
                    
                    keypoints.extend(kp)
                    if des is not None:
                        if descriptors is None:
                            descriptors = des
                        else:
                            descriptors = np.vstack((descriptors, des))
        keypoints = np.array(keypoints)
        return keypoints, descriptors
        
    
class FeatureMatcher(object):

    def __init__(self, K, W, H):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.display = Display(W, H)

    def match(self, frame, last):
        idx1 = []
        idx2 = []

        # feature matching
        idxFrame, idxLast = self.matchFeatures(frame, last)

        # filtering matches using ransac and essential matrix
        model, inliers = ransac(
            (frame.pts[idxFrame], last.pts[idxLast]),
            # FundamentalMatrixTransform,
            EssentialMatrixTransform,
            min_samples=8,
            residual_threshold=0.04,
            max_trials=200,
        )
        print(f'{inliers.sum()} inliers')
        idxFrame = idxFrame[inliers]
        idxLast = idxLast[inliers]
        self.showKeypointsAndMatches(frame, frame.pts[idxFrame], last.pts[idxLast])

        Rt = extractRt(model.params)
 
        return idxFrame, idxLast, Rt

    def matchFeatures(self, frame, last):

        idxFrame = []
        idxLast = []
        matches = self.bf.knnMatch(frame.des, last.des, k=2)                   
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                idxFrame.append(m.queryIdx)
                idxLast.append(m.trainIdx)
        assert len(idxFrame) > 8
        idxFrame = np.array(idxFrame)
        idxLast = np.array(idxLast)
        
        return idxFrame, idxLast


    def denormalize_point(self, pt):
        """
        Denormalizes assingle point using the camera intrinsic matrix.

        Parameters
        ----------
        pt : tuple
            The point to denormalize.

        Returns
        -------
        denormalized_point : tuple
            The denormalized point.
        """
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        denormalized_point =  int(round(ret[0])), int(round(ret[1]))
        return denormalized_point
    
    def showKeypointsAndMatches(self, frame, frame_inliers, last_inliers):
        """
        Displays the keypoints and matches on the image.

        Parameters
        ----------
        frame : frame.Frame
            The frame to show the keypoints and matches on.
        matches : list
            The list of matched keypoints.

        Returns
        -------
        None
        """
        for (pt1, pt2) in zip(frame_inliers, last_inliers):
            u1, v1 = map(int, self.denormalize_point(pt1))
            u2, v2 = map(int, self.denormalize_point(pt2))
            cv2.circle(frame.img, (u1, v1), 2, (0, 255,0), 1)
            cv2.line(frame.img, (u1, v1), (u2, v2), (255, 0, 0), 1)
        self.display.show(frame.img) 

