#!/usr/bin/env python3
import cv2 
from frame import FeatureExtractorMatcher

feature_extractor = FeatureExtractorMatcher()

def process_image(img):
    kp, des = feature_extractor.extract(img, method='shitomasi')
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
