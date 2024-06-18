import cv2
import numpy as np
from display import Display

class FeatureExtractor(object):

    def __init__(self):
        self.orb = cv2.ORB_create(500)
        self.display = Display()

    def extract(self, img, method = 'orb'):
        # extracts the features for matching
        if method == 'orb':
            kp, des = self.orb_extract(img)

        elif method == 'shitomasi':
            kp, des = self.shiTomasi_extract(img)

        self.showKeypoints(img, kp)
        return kp, des

    def shiTomasi_extract(self, img):
        # Shi Tomasi corner detection
        keypoints = []
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_img, 3000, 0.01, 7)
        keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in corners]
        
        # compute orb description for the shi tomasi corners
        keypoints, descriptors = self.orb.compute(img, keypoints)
        
        return keypoints, descriptors

    def orb_extract(self, img):
        # orb feature extraction using grid based technique
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

    def showKeypoints(self, img, kp):
        img = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        self.display.show(img)