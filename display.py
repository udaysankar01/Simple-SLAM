import cv2
from utils import denormalize_point

class Display(object):
    """
    Display class to show the image in a window.

    Parameters
    ----------
    W : int
        The width of the window.
    H : int
        The height of the window.

    Returns
    -------
    None
    """
    def __init__(self, W, H):
        self.W = W
        self.H = H

    def show(self, img):
        """
        Displays the image in a window.

        Parameters
        ----------
        img : np.array
            The image to display.

        Returns
        -------
        None
        """
        cv2.imshow('frame', img)
        cv2.moveWindow('frame', 1200, 50)

    def showKeypointsAndMatches(self, slam_map, idx1, idx2):
        """
        Displays the keypoints and matches on the image.

        Parameters
        ----------
        frame : frame.Frame
            The frame to display the keypoints and matches on.
        frame_inliers : np.array
            The inliers of the current frame.
        last_inliers : np.array
            The inliers of the last frame.

        Returns
        -------
        None
        """
        
        frame = slam_map.frames[-1]
        frame_inliers = slam_map.frames[-1].keypoints[idx1]
        last_inliers = slam_map.frames[-2].keypoints[idx2]
        for (pt1, pt2) in zip(frame_inliers, last_inliers):
            u1, v1 = map(int, denormalize_point(pt1, frame.K))
            u2, v2 = map(int, denormalize_point(pt2, frame.K))
            cv2.circle(frame.img, (u1, v1), 2, (0, 255,0), 1)
            cv2.line(frame.img, (u1, v1), (u2, v2), (255, 0, 0), 1)
        self.show(frame.img) 
        