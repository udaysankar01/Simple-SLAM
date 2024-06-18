# Simple-SLAM

A simple implementation of SLAM (Simultaneous Localization and Mapping) using Python.

## Features

- [x] Feature detection using ORB feature detection or Shi-Tomasi corner detection.
  - [ ] ORB features are clustered -- ANMS?
- [x] Feature matching using Brute Force Matching (knn matching -- no cross checks).
  - [x] Clean up matches using ratio test -- takes care of most outlier matches.
  - [ ] Clean up remaining matches using alternative methods -- Fundamental Matrix?

## TODO

- Implement alternative methods for cleaning up matches.
- Integrate more advanced SLAM techniques.
- Optimize performance for real-time processing.

## Usage

1. **Feature Detection:**

   - ORB feature detection
   - Shi-Tomasi corner detection

2. **Feature Matching:**
   - Brute Force Matching
   - Ratio test for cleaning up matches

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

To run the program:

```sh
./slam.py
```
