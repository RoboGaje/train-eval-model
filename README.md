# Face and Body Detection/Recognition System

A comprehensive machine learning system for face detection, face recognition, and body detection using state-of-the-art deep learning models. This system combines YOLO for object detection and FaceNet for face recognition to provide accurate and efficient real-time processing capabilities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Performance](#model-performance)
- [Directory Structure](#directory-structure)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Technical Details](#technical-details)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a dual-model architecture for comprehensive face and body analysis:

1. **Face Detection**: YOLO models fine-tuned for face detection
2. **Face Recognition**: FaceNet model for identity recognition
3. **Body Detection**: YOLO pre-trained models for human body detection
4. **Crowd Analysis**: Automated crowd density analysis and statistics

The system supports both real-time processing and batch processing of images and videos, with TensorRT optimization for production deployment.

## Features

### Core Capabilities

- **Dual Detection System**: Simultaneous face and body detection
- **Face Recognition**: Identity recognition with 96.36% accuracy
- **Real-time Processing**: Live webcam and video processing
- **TensorRT Optimization**: 3-5x performance improvement for inference
- **Crowd Analysis**: Automated crowd density classification and statistics
- **Multi-format Support**: Images, videos, and real-time streams

### Advanced Features

- **Hybrid Prediction**: Combined YOLO detection and FaceNet recognition
- **Smart Frame Processing**: Configurable frame intervals for optimal performance
- **Crowd Level Classification**: Empty, Low, Medium, High, Very High
- **Statistics Export**: Comprehensive analytics export to JSON/CSV
- **Model Comparison**: Built-in evaluation and benchmarking tools

## System Architecture

```
Input (Image/Video/Stream)
           ↓
    YOLO Detection
    (Face & Body Bounding Boxes)
           ↓
      Face Cropping
           ↓
    FaceNet Recognition
    (Identity Classification)
           ↓
    Combined Output
    (Detection + Recognition + Statistics)
```

### Supported Classes

**Face Recognition Classes:**

- `dimas` - Specific individual
- `fabian` - Specific individual
- `sendy` - Specific individual
- `syahrul` - Specific individual
- `unknown` - Unrecognized faces

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- TensorRT 8.0+ (optional, for optimization)

### Dependencies Installation

```bash
# Clone the repository
git clone https://github.com/RoboGaje/train-eval-model.git
cd face-body-detection-system

# Install required packages
pip install -r requirements.txt
```

### Model Setup

1. **Download Pre-trained Models:**

   - YOLO models are included in the repository
   - FaceNet models: Download from [Google Drive](https://drive.google.com/drive/folders/1cJa4RmXEZ1U0oLc0kgoJ_MnNOemxInh7?usp=sharing)
2. **Place models in correct directories:**

   ```
   models/
   ├── YOLO12n_finetuned/weights/best.pt
   ├── facenet_models/best_facenet.pth
   └── facenet_models/class_mapping.pkl
   ```

## Quick Start

### Dataset Verification

```bash
python check_dataset.py
```

### Basic Image Processing

```bash
python inference/facenet_yolo_inference.py \
  --yolo-model models/YOLO12n_finetuned/weights/best.pt \
  --facenet-model models/facenet_models/best_facenet.pth \
  --mode image \
  --input test/images/sample.jpg \
  --show
```

### Video Processing with TensorRT

```bash
python inference/facenet_yolo_inference.py \
  --yolo-model models/YOLO12n_finetuned/weights/best.engine \
  --facenet-model models/facenet_models/best_facenet.pth \
  --mode video \
  --input video.mp4 \
  --use-tensorrt \
  --output result.mp4
```

### Real-time Webcam Processing

```bash
python inference/realtime_tracking.py \
  --yolo-model models/YOLO12n_finetuned/weights/best.pt \
  --facenet-model models/facenet_models/best_facenet.pth
```

## Model Performance

### FaceNet Recognition Performance

- **Overall Accuracy**: 96.36%
- **Test Loss**: 0.0798

**Per-Class Accuracy:**

| Class   | Accuracy |
| ------- | -------- |
| dimas   | 71.43%   |
| fabian  | 100.00%  |
| sendy   | 95.45%   |
| syahrul | 100.00%  |
| unknown | 96.71%   |

### YOLO Model Comparison

| Model    | Size (MB) | mAP@0.5 | mAP@0.5:0.95 | Inference Time | Efficiency |
| -------- | --------- | ------- | ------------ | -------------- | ---------- |
| YOLOv12n | 5.26      | 94.47%  | 83.02%       | Fastest        | Highest    |
| YOLOv12s | 18.05     | 96.42%  | 81.65%       | Fast           | High       |
| YOLOv12m | 38.88     | 96.44%  | 85.73%       | Medium         | Medium     |
| YOLOv12l | 51.05     | 94.90%  | 85.33%       | Slow           | Low        |
| YOLOv12x | 113.58    | 96.12%  | 86.86%       | Slowest        | Lowest     |

**Recommendation**: YOLOv12n provides the best balance between performance and efficiency.

## Directory Structure

```
train-eval-model-raw-main/
├── check_dataset.py              # Dataset verification script
├── data.yaml                     # Dataset configuration
├── requirements.txt              # Python dependencies
├── demo_output_images/           # Sample output images
├── evaluation/                   # Model evaluation scripts
│   ├── evaluate_all_models.py    # Comprehensive model evaluation
│   ├── benchmark_runtime.py      # Performance benchmarking
│   └── model_evaluation_results.csv
├── inference/                    # Inference scripts
│   ├── facenet_yolo_inference.py # Main hybrid inference script
│   ├── face_body_detection_inference.py # Dual detection script
│   ├── realtime_tracking.py      # Real-time processing
│   ├── demo_tracking.py          # Demo with file output
│   └── README.md                 # Inference documentation
├── models/                       # Trained models
│   ├── YOLO12n_finetuned/       # Fine-tuned YOLO models
│   └── facenet_models/          # FaceNet models and mappings
└── training/                     # Training scripts
    ├── train_yolo_models.py      # YOLO training script
    ├── train_facenet.py          # FaceNet training script
    ├── preprocess_for_facenet.py # Dataset preprocessing
    └── README.md                 # Training documentation
```

## Training

### Dataset Preprocessing

Convert YOLO dataset to FaceNet format:

```bash
cd training
python preprocess_for_facenet.py \
  --data ../data.yaml \
  --output ../facenet_dataset \
  --padding 0.2
```

### FaceNet Training

Train FaceNet model with timestamp-based output:

```bash
python train_facenet.py \
  --dataset ../facenet_dataset \
  --epochs 50 \
  --batch-size 602 \
  --lr 0.001
```

### YOLO Fine-tuning

Train and compare multiple YOLO variants:

```bash
python train_yolo_models.py
```

## Inference

### Face + Body Detection with Crowd Analysis

```bash
python inference/face_body_detection_inference.py \
  --face-model models/YOLO12n_finetuned/weights/best.pt \
  --body-model yolo12n.pt \
  --mode video \
  --input video.mp4 \
  --save-stats \
  --face-conf 0.5 \
  --body-conf 0.5
```

### Parameters Reference

| Parameter            | Default  | Description                          |
| -------------------- | -------- | ------------------------------------ |
| `--yolo-model`     | Required | Path to YOLO model (.pt or .engine)  |
| `--facenet-model`  | Required | Path to FaceNet model (.pth)         |
| `--mode`           | image    | Processing mode: image or video      |
| `--input`          | Required | Input file path                      |
| `--output`         | Optional | Output file path                     |
| `--yolo-conf`      | 0.5      | YOLO confidence threshold            |
| `--facenet-conf`   | 0.7      | FaceNet confidence threshold         |
| `--use-tensorrt`   | False    | Enable TensorRT acceleration         |
| `--frame-interval` | 3        | Frame processing interval for videos |
| `--show`           | False    | Display results in real-time         |

## Evaluation

### Comprehensive Model Evaluation

```bash
python evaluation/evaluate_all_models.py
```

### Runtime Benchmarking

```bash
python evaluation/benchmark_runtime.py \
  --model models/YOLO12n_finetuned/weights/best.pt \
  --engine models/YOLO12n_finetuned/weights/best.engine
```

## Technical Details

### Model Architectures

**YOLO Configuration:**

- Base models: YOLOv12n/s/m/l/x
- Fine-tuned on custom face dataset
- 5 classes: dimas, fabian, people face, sendy, syahrul
- Input resolution: 640x640

**FaceNet Configuration:**

- Backbone: InceptionResnetV1
- Feature dimension: 512
- Hidden layers: 256
- Dropout rates: 0.5, 0.3
- Input resolution: 160x160

### Performance Optimization

**TensorRT Integration:**

- Automatic model conversion to TensorRT engine
- 3-5x inference speed improvement
- Automatic fallback to PyTorch models
- Memory optimization for edge deployment

**Smart Processing:**

- Configurable frame intervals for video processing
- Memory-efficient batch processing
- GPU memory management and optimization

### Crowd Analysis Metrics

**Density Classification:**

- Empty: 0 persons
- Low: 1-2 persons
- Medium: 3-5 persons
- High: 6-10 persons
- Very High: >10 persons

**Statistics Export:**

- Per-frame detection counts
- Average crowd density
- Processing performance metrics
- Confidence score distributions

## Configuration

### Dataset Configuration (data.yaml)

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
nc: 5
names: ["dimas", "fabian", "people face", "sendy", "syahrul"]
```

### Training Configuration

```python
# YOLO Training
EPOCHS = 50
BATCH_SIZES = {
    'YOLOv12n': 100,
    'YOLOv12s': 55,
    'YOLOv12m': 30,
    'YOLOv12l': 20,
    'YOLOv12x': 13
}

# FaceNet Training
EPOCHS = 50
BATCH_SIZE = 602
LEARNING_RATE = 0.001
FEATURE_DIM = 512
HIDDEN_DIM = 256
```

## System Requirements

### Minimum Requirements

- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8GB
- Storage: 10GB free space
- Python: 3.8+

### Recommended for Production

- CPU: Intel i7/i9 or AMD Ryzen 7/9
- GPU: NVIDIA RTX 3060 or better (8GB+ VRAM)
- RAM: 16GB+
- Storage: 50GB+ SSD
- CUDA: 11.0+
- TensorRT: 8.0+

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation for API changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO (You Only Look Once) by Ultralytics
- FaceNet by Google Research
- PyTorch Deep Learning Framework
- TensorRT by NVIDIA

## Support

For issues and questions:

1. Check the documentation in respective README files
2. Search existing issues in the repository
3. Create a new issue with detailed description and logs
4. Include system specifications and model versions
