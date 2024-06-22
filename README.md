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

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

To run the program:

```sh
./slam.py
```
