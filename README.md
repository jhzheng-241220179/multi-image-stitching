# Robust Multi-Image Stitching & Panorama Generation

A computer vision project for automatic pairwise and multi-image panorama generation.

## Features

- SIFT feature extraction
- FLANN feature matching
- Lowe's ratio test
- RANSAC homography estimation
- Image matching graph
- Global homography propagation
- Perspective warping
- Weighted image blending

## Tech Stack

Python · OpenCV · NumPy · NetworkX

## Files

- `two_image_stitching.py` — pairwise image stitching
- `multi_image_stitching.py` — multi-image panorama generation
- `cv_report.pdf` — project report
- `requirements.txt` — dependencies

## Run

```bash
pip install -r requirements.txt
python two_image_stitching.py
Author

Jinghan Zheng
Nanjing University
