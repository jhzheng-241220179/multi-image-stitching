# Robust Multi-Image Stitching & Panorama Generation

Automatic pairwise and multi-image panorama generation based on feature matching and homography estimation.

## Features

- SIFT feature extraction
- FLANN feature matching
- RANSAC homography estimation
- Image matching graph
- Global image alignment
- Perspective warping
- Weighted blending

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
python multi_image_stitching.py
