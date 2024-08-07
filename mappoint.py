import numpy as np
from multiprocessing import Process, Queue

import g2o
import pangolin
import OpenGL.GL as gl

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = None

    # ---- Optimizer ----
    def optimize(self):

        # optimizer initialization
        opt = g2o.SparseOptimizer()
        
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
        opt.set_algorithm(solver)

        sbacam = g2o.SBACam()
        # sbacam.set_cam(1.0, 1.0, 0.0, 0.0, 0)   # camera parameters: fx, fy, cx, cy, baseline
        sbacam.set_cam(self.frames[0].K[0][0], self.frames[0].K[1][1], self.frames[0].K[0][2], self.frames[0].K[1][2], 1.0)

        # add frames to the graph
        for frame in self.frames:
            Rt = frame.Rt
            quaternion = g2o.Quaternion(Rt[:3, :3])
            sbacam.set_rotation(quaternion)
            sbacam.set_translation(Rt[:3, 3])

            v_cam = g2o.VertexCam()
            v_cam.set_id(frame.id)
            v_cam.set_estimate(sbacam)
            v_cam.set_fixed(frame.id <= 1) # fixing first two frames to constrain the scale
            opt.add_vertex(v_cam)

        # add points to the graph
        pt_index_offset = 0x10000
        for point in self.points:
            v_point = g2o.VertexPointXYZ()
            v_point.set_id(point.id + 0x10000)
            v_point.set_estimate(point.position[:3])
            v_point.set_marginalized(True)
            v_point.set_fixed(False)
            opt.add_vertex(v_point)

            for i, frame in enumerate(point.frames):
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, v_point)
                edge.set_vertex(1, opt.vertex(frame.id))
                edge.set_measurement(frame.keypoints_unnorm[frame.pts.index(point)])
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)

        opt.initialize_optimization()
        opt.set_verbose(True)
        opt.optimize(10)

        # add frame poses back into map
        for frame in self.frames:
            est = opt.vertex(frame.id).estimate()
            frame.Rt = est.to_homogeneous_matrix()
        
        # add points to the map
        for point in self.points:
            est = opt.vertex(point.id + pt_index_offset).estimate()
            point.position = np.array([est[0], est[1], est[2], 1.0])


    # ------ Viewer ------

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
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        # camera Rt
        gl.glPointSize(10)
        gl.glColor(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0])
        
        # point position
        gl.glPointSize(5)
        pangolin.DrawPoints(np.array(self.state[1]), self.state[2])   

        pangolin.FinishFrame()


    def display(self):
        if self.q is None:
            return
        Rts = np.array([frame.Rt for frame in self.frames])
        pts = np.array([point.position for point in self.points])
        colors = np.array([point.color for point in self.points])
        colors = colors / 255.0
        self.q.put((Rts, pts, colors))


class Point(object):
    
    def __init__(self, slam_map, position, pointColor):
        self.frames = []
        self.position = position
        self.idxs = []
        self.color = pointColor.copy()
        self.id = len(slam_map.points)
        slam_map.points.append(self)

    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
