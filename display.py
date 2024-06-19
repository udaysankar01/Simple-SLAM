import cv2

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
        img = cv2.resize(img, (self.W, self.H))
        cv2.imshow('frame', img)