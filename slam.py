#!/usr/bin/env python3
import cv2 
import numpy as np
from feature import FeatureExtractorMatcher

W = 1920//2
H = 1080//2
F = 210
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

feature_extractor = FeatureExtractorMatcher(K, W, H)

def process_image(img):
    macthes = feature_extractor.extract(img, method='shitomasi')
    return img

def main():
    cap = cv2.VideoCapture('videos/test.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = process_image(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
