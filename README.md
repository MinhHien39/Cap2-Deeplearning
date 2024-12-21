# Crime Detection API

A FastAPI-based service for detecting criminal activities in videos and images using computer vision and deep learning.

## Features

- Real-time crime detection in videos and images
- Support for multiple crime categories:
  - Vandalism
  - Shooting
  - Explosion
  - Arrest
  - Assault
  - Fighting
  - Road accidents
  - Robbery
- Custom confidence thresholds for each crime type
- Processed media storage and retrieval
- ZIP archive generation for processed results
- REST API endpoints for video/image processing

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda package manager
- Required Python packages:

```
fastapi==0.104.1
python-multipart==0.0.6
uvicorn==0.24.0
opencv-python==4.8.1.78
numpy==1.26.0
python-jose==3.3.0
python-dotenv==1.0.0
ultralytics
```
