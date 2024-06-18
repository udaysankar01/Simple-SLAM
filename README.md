# Simple-SLAM

A simple implementation of SLAM (Simultaneous Localization and Mapping) using Python.

## Features

- [x] Feature detection using ORB feature detection or Shi-Tomasi corner detection.
  - [ ] ORB features are clustered -- ANMS?
- [x] Feature matching using Brute Force Matching (knn matching -- no cross checks).
  - [x] Clean up matches using ratio test -- takes care of most outlier matches.
  - [x] Clean up remaining matches using alternative methods -- ransac with Essential Matrix

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
   - Ransac with Essential Matrix to remove outlier matches

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

To run the program:

```sh
./slam.py
```
