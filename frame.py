import cv2
import math
import numpy as np
from display import Display

display = Display()

class FeatureExtractorMatcher(object):

    def __init__(self):
        self.orb = cv2.ORB_create(500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last = None

    def extract(self, img, method = 'orb'):
        # feature extraction
        if method == 'orb':
            kps, des = self.extractOrb(img)
        elif method == 'shitomasi':
            kps, des = self.extractShiTomasi(img)

        # feature matching
        if self.last is not None:
            matches = self.matchFeatures(img, (kps, des), self.last)

            # ratio test
            good_matches = []
            for i, (pt1, pt2) in enumerate(matches):
                if math.dist(pt1, pt2) < 100:
                    good_matches.append((pt1, pt2))

            self.showKeypointsAndMacthes(img, kps, good_matches)
        self.last = (kps, des)                                      ####### IS DICTIONARY BETTER???? 

        return kps, des


    def extractShiTomasi(self, img):
        # Shi Tomasi corner detection
        keypoints = []
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_img, 3000, 0.01, 7)
        keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in corners]
        
        # compute orb description for the shi tomasi corners
        keypoints, descriptors = self.orb.compute(img, keypoints)
        
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
        
        return keypoints, descriptors
    

    def matchFeatures(self, img, current, last):      ############## CLEAN UP
        """
        Matches the features between the current and last frame.
        
        Parameters
        ----------
        img : np.array
            The current frame.
        current : tuple
            The keypoints and descriptors of the current frame.
        last : tuple
            The keypoints and descriptors of the last frame.

        Returns
        -------
        matches : list
            List of matched keypoints.
        """
        kp1, des1 = last
        kp2, des2 = current
        matches = self.bf.match(des1, des2)                   
        print(len(matches), "matches.")
        
        def get_coords(match):
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            return (pt1, pt2)
        
        matches = list(map(get_coords, matches))
        return matches

    
    def showKeypointsAndMacthes(self, img, kps, matches):
        # plotting keypoints on the image
        img = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)

        # plotting the matches on the image
        for (pt1, pt2) in matches:
            cv2.line(img, pt1, pt2, (255, 0, 0), 1)
        display.show(img) 