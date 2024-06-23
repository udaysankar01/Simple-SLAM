import cv2
import numpy as np
from display import Display
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
from utils import *
np.set_printoptions(suppress=True)

focal_length_estimation_flag = False # not tested yet!
focal_lengths = []

class FeatureExtractor(object):

    def __init__(self, method='shitomasi'):
        self.supported_methods = ['orb', 'shitomasi']
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported method: {method}. Choose method from {self.supported_methods} ")
        self.method = method
        self.grid_size = (4, 4)
        self.orb = cv2.ORB_create(500)
    
    def extract(self, img):
        """
        Extracts features from the image.

        Parameters
        ----------
        img : np.array
            The image to extract features from.

        Returns
        -------
        kps : np.array
            The keypoints.
        des : np.array
            The descriptors of the keypoints.
        """
        if self.method == 'orb':
            kps, des = self.extractOrb(img)
        elif self.method == 'shitomasi':
            kps, des = self.extractShiTomasi(img, n_points=1000)
        
        return kps, des
    
    def keypointsToPoints(self, keypoints):
        """
        Converts the keypoints to points.

        Parameters
        ----------
        keypoints : np.array
            The OpenCV keypoints to convert.

        Returns
        -------
        points : np.array
            The points.
        """
        points = []
        for keypoint in keypoints:
            points.append(keypoint.pt)
        points = np.array(points)
        return points
    
    def extractShiTomasi(self, img, n_points=1000):
        """
        Extracts Shi Tomasi features from the image.

        Parameters
        ----------
        img : np.array
            The image to extract features from.

        Returns
        -------
        keypoints : np.array
            Array of keypoints.
        descriptors : np.array
            The descriptors of the keypoints.
        """
        features = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), n_points, 0.01, 10)
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

    def match(self, frame1, frame2):
        """
        Matches the features of the frames and returns the inliers and the projection matrix.

        Parameters
        ----------
        frame1 : frame.Frame
            The first frame.
        frame2 : frame.Frame
            The second frame.

        Returns
        -------
        idx1 : np.array
            The indices of the inliers in the first frame.
        idx2 : np.array
            The indices of the inliers in the second frame.
        Rt : np.array
            The 3x4 projection matrix.
        """
        # feature matching
        idx1, idx2 = self.matchFeatures(frame1, frame2)
        frame1_pts = frame1.pts[idx1].copy()
        frame2_pts = frame2.pts[idx2].copy()

        # ransac hyperparameters
        model_class = EssentialMatrixTransform
        min_samples = 8
        residual_threshold = 0.04
        max_trials = 100
        
        if focal_length_estimation_flag:
            frame1_pts = frame1.pts_unnorm[idx1].copy()
            frame2_pts = frame2.pts_unnorm[idx2].copy()
            model_class = FundamentalMatrixTransform
            residual_threshold = 1
            max_trials = 1000

        # filtering matches using ransac and essential matrix
        model, inliers = ransac(
            (frame1_pts, frame2_pts),
            model_class,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )
        if focal_length_estimation_flag:
            
            _, D, _ = np.linalg.svd(model.params)
            print(D)
            
        # print(f'{inliers.sum()} inliers')
        idx1 = idx1[inliers]
        idx2 = idx2[inliers]

        Rt = extractRt(model.params)
 
        return idx1, idx2, Rt

    def matchFeatures(self, frame1, frame2):
        """
        Matches the features of the frames.

        Parameters
        ----------
        frame1 : frame.Frame
            The first frame.
        frame2 : frame.Frame
            The second frame.

        Returns
        -------
        idx1 : np.array
            The indices of the inliers in the first frame.
        idx2 : np.array
            The indices of the inliers in the second frame.
        """
        idx1 = []
        idx2 = []
        matches = self.bf.knnMatch(frame1.des, frame2.des, k=2)                   
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                p1 = frame1.pts_unnorm[m.queryIdx]
                p2 = frame2.pts_unnorm[m.trainIdx]
                if np.linalg.norm(p1 - p2) < 0.1*frame1.W:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
        assert len(idx1) > 8
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        
        return idx1, idx2
