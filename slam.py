#!/usr/bin/env python
import cv2 
import numpy as np
from feature import FeatureExtractor, FeatureMatcher
from frame import Frame
import g2o
from utils import *

from multiprocessing import Process, Queue
import pangolin
import OpenGL.GL as gl

W = 1920//2
H = 1080//2
F = 280
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

feature_extractor = FeatureExtractor(method='shitomasi')
feature_matcher = FeatureMatcher(K, W, H)

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []

        # create viewer process
        self.q = Queue()
        self.viewer_process = Process(target=self.viewer_thread, args=(self.q,))
        self.viewer_process.daemon = True
        self.state = None
        self.viewer_process.start()

    def viewer_thread(self, q):
        self.viewerInit()

        self.state = None
        flag = True
        while not pangolin.ShouldQuit():
            self.viewerUpdate(q)

    
    def viewerInit(self):
        pangolin.CreateWindowAndBind('SLAM', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    def viewerUpdate(self, q):
        if self.state is None or not q.empty():
                self.state = q.get()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        gl.glPointSize(10)
        gl.glColor(0.0, 1.0, 0.0)
        pangolin.DrawPoints(np.array([d[:3, 3] for d in self.state[0]]))
                
        gl.glPointSize(2)
        gl.glColor(0.0, 1.0, 0.0)
        pangolin.DrawPoints(np.array(self.state[1]))   

        pangolin.FinishFrame()

    def stop_viewer(self):
        self.viewer_process.terminate()
        self.viewer_process.join()

    def display(self):
        Rts = [frame.Rt for frame in self.frames]
        pts = [point.position for point in self.points]
        self.q.put((Rts, pts))


slam_map = Map()    

class Point(object):
    
    def __init__(self, slam_map, position):
        self.frames = []
        self.position = position
        self.idxs = []
        
        self.id = len(slam_map.points)
        slam_map.points.append(self)


    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)



def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def process_image(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(slam_map, img, K, feature_extractor)
    if frame.id == 0:
        return
    
    idx1, idx2, slam_map.frames[-1].Rt = feature_matcher.match(slam_map.frames[-1], slam_map.frames[-2])

    # triangulate points
    slam_map.frames[-1].Rt = np.dot(slam_map.frames[-2].Rt, slam_map.frames[-1].Rt)
    pts4d = triangulate(slam_map.frames[-1].Rt, slam_map.frames[-2].Rt, slam_map.frames[-1].pts[idx1], slam_map.frames[-2].pts[idx2])

    # reject unwanted points
    filter_pts3d_index = np.abs(pts4d[:, 3]) > 0.005
    pts4d = pts4d[filter_pts3d_index]
    pts4d /= pts4d[:, 3].reshape(-1, 1)
    pts4d = pts4d[pts4d[:, 2] > 0]

    for p in pts4d:
        pt = Point(slam_map, p)
        pt.add_observation(slam_map.frames[-1], idx1)
        pt.add_observation(slam_map.frames[-2], idx2)
    
    slam_map.display()


def main():
    cap = cv2.VideoCapture('videos/test.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            process_image(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else:
            break
    

    cap.release()
    cv2.destroyAllWindows()
    slam_map.stop_viewer()
    

if __name__ == '__main__':
    main()
