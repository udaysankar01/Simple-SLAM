# Simple-SLAM

A simple implementation of sparse SLAM (Simultaneous Localization and Mapping) using Python.

## Tasks

- [x] Feature detection using ORB feature detection or Shi-Tomasi corner detection.
  - [ ] ORB features are clustered -- ANMS?
- [x] Feature matching using Brute Force Matching (knn matching -- no cross checks).
  - [x] Clean up matches using ratio test -- takes care of most outlier matches.
  - [x] Clean up remaining matches using alternative methods -- RANSAC with Essential Matrix.
- [x] Get rotation and translation from Essential Matrix.
  - [x] Fix the error in translation vector.
- [x] Triangulate 3D points from the matched features.
  - [x] Implement a Map class to store the 3D points.
  - [x] Build and visualize the 3D map using Pangolin.
  - [x] Fix Bug: Camera Drift in XY Plane during Slow Movement --> caused by error in intrinsic matrix.
- [ ] Tracking of points over multiple frames.
- [ ] g2o based optimization.
- [ ] point culling.

## TODO

- Integrate more advanced SLAM features.
- Optimize performance for real-time processing.

## Usage

1. **Feature Detection:**

   - ORB feature detection
   - Shi-Tomasi corner detection

2. **Feature Matching:**

   - Brute Force Matching
   - Ratio test for cleaning up matches
   - RANSAC with Essential Matrix to remove outlier matches

3. **Rotation and Translation Estimation:**

   - Estimation of rotation and translation from the Essential Matrix.

4. **Triangulation:**
   - Triangulation of 3D points using Direct Linear Transform (DLT).

## Installation

This project is coded using Python 3.10.14.

This project requires the Python binding for Pangolin. The Pangolin library is a lightweight portable rapid development library for managing OpenGL display / interaction and abstracting video input. To install Pangolin, use this repo:

https://github.com/uoip/pangolin

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

To run the program:

```sh
D2D=1 D3D=1 ./slam.py --video <video_file>
```
