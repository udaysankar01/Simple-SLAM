import numpy as np
from scipy.spatial import KDTree

import pangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = None
        # test
        self.kdtree = None
        self.positions = [] # list to store positions for kdtree

    def updateKDTree(self):
        if self.positions:
            self.kdtree = KDTree(np.array(self.positions))

    def addOrUpdatePoint(self, new_pt):
        if self.kdtree is not None:
            dist, idx = self.kdtree.query(new_pt, distance_upper_bound=0.1)
            if dist < float('inf'):
                existing_point = self.points[idx]
                print(new_pt, existing_point.position)
                return existing_point
        new_point = Point(self, new_pt)
        self.positions.append(new_pt)
        self.updateKDTree()
        return new_point

    def create_viewer(self):
        self.q = Queue()
        self.viewer_process = Process(target=self.viewer_thread, args=(self.q,))
        self.viewer_process.daemon = True
        self.viewer_process.start()

    def viewer_thread(self, q):
        self.viewerInit(1024, 768)
        self.state = None
        flag = True
        while not pangolin.ShouldQuit():
            self.viewerUpdate(q)

    
    def viewerInit(self, w, h):
        pangolin.CreateWindowAndBind('SLAM', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8, 
                                     0, 0, 0, 
                                     0, -0.1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w//h)
        self.dcam.SetHandler(self.handler)

    def viewerUpdate(self, q):
        if self.state is None or not q.empty():
                self.state = q.get()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # camera Rt
        gl.glPointSize(10)
        gl.glColor(0.0, 1.0, 0.0)
        # pangolin.DrawCameras(self.state[0], w=1.0, h_ratio=0.5625, z_ratio=1)
        pangolin.DrawCameras(self.state[0])
        
        # point position
        gl.glPointSize(2)
        gl.glColor(1.0, 0.0, 0.0)
        pangolin.DrawPoints(np.array(self.state[1]))   

        pangolin.FinishFrame()

    def stop_viewer(self):
        self.viewer_process.terminate()
        self.viewer_process.join()

    def display(self):
        if self.q is None:
            return
        Rts = [frame.Rt for frame in self.frames]
        pts = [point.position for point in self.points]
        self.q.put((Rts, pts))


class Point(object):
    
    def __init__(self, slam_map, position):
        self.frames = []
        self.position = position
        self.idxs = []
        
        self.id = len(slam_map.points)
        slam_map.points.append(self)

    def add_observation(self, frame, idx):
        frame.kps[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)