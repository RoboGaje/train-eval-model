# 🚀 YOLO Model Training & Real-time Tracking

Repository ini berisi implementasi lengkap untuk fine-tuning model YOLO dan FaceNet, serta real-time object tracking dengan berbagai runtime engine.

## 📋 Daftar Isi

- [Dataset](#-dataset)
- [Model yang Tersedia](#-model-yang-tersedia)
- [Struktur Direktori](#-struktur-direktori)
- [Script yang Tersedia](#-script-yang-tersedia)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Hasil Benchmark](#-hasil-benchmark)
- [Troubleshooting](#-troubleshooting)

## 📊 Dataset

**Face Detection Dataset** dengan 5 kelas:
- `dimas` - Wajah Dimas
- `fabian` - Wajah Fabian  
- `people face` - Wajah umum
- `sendy` - Wajah Sendy
- `syahrul` - Wajah Syahrul

**Statistik Dataset:**
- Total: 2,973 gambar
- Training: 2,871 gambar
- Validation: 52 gambar
- Test: 50 gambar

## 🤖 Model yang Tersedia

### YOLO Models
| Model | Ukuran File | Runtime | Lokasi |
|-------|-------------|---------|--------|
| YOLOv12n | 5.3MB | PyTorch | `models/YOLO12n/weights/best.pt` |
| YOLOv12n | 19MB | TensorRT | `models/YOLO12n/weights/best.engine` |
| YOLOv12s | 54MB | PyTorch | `models/YOLO12s/weights/best.pt` |
| YOLOv12m | ~39MB | PyTorch | `models/YOLOv12m/weights/best.pt` |
| YOLOv12l | ~51MB | PyTorch | `models/YOLOv12l/weights/best.pt` |
| YOLOv12x | ~114MB | PyTorch | `models/YOLOv12x/weights/best.pt` |

### FaceNet Models
| Model | Accuracy | Lokasi |
|-------|----------|--------|
| FaceNet Best | 98.50% | `models/facenet_models/latest/best_facenet.pth` |
| Class Mapping | - | `models/facenet_models/latest/class_mapping.pkl` |

## 📁 Struktur Direktori

```
robotika-uas/
├── 📁 inference/                    # 🚀 Script inference (UTAMA)
│   ├── facenet_yolo_inference.py    # YOLO + FaceNet inference
│   ├── facenet_inference.py         # FaceNet pure inference
│   ├── demo_tracking.py             # Demo tracking
│   ├── realtime_tracking.py         # Real-time webcam
│   └── README.md                    # Dokumentasi inference
├── 📁 models/                       # Model yang sudah dilatih
│   ├── YOLO12n/weights/
│   ├── YOLO12s/weights/
│   ├── facenet_models/
│   │   ├── latest/                  # Symlink ke training terbaru
│   │   └── training_YYYYMMDD_HHMMSS/
├── 📁 evaluation/                   # Script evaluasi
├── 📁 facenet_dataset/              # Dataset FaceNet
├── 📁 test/images/                  # Test images
├── train_yolo_models.py             # Training YOLO
├── train_facenet.py                 # Training FaceNet
├── preprocess_for_facenet.py        # Preprocessing FaceNet
└── check_dataset.py                 # Dataset checker
```

## 📜 Script yang Tersedia

### 🎯 Inference Scripts (Direktori `inference/`)
- **`facenet_yolo_inference.py`** - **SCRIPT UTAMA**: YOLO + FaceNet inference
- **`facenet_inference.py`** - FaceNet pure inference dengan MTCNN
- **`demo_tracking.py`** - Demo tracking dengan output file
- **`realtime_tracking.py`** - Real-time tracking dengan webcam

### 🏋️ Training Scripts
- **`train_yolo_models.py`** - Fine-tuning semua model YOLO
- **`train_facenet.py`** - Fine-tuning FaceNet dengan timestamp folders
- **`preprocess_for_facenet.py`** - Preprocessing dataset untuk FaceNet

### 🔍 Utility Scripts
- **`check_dataset.py`** - Verifikasi struktur dan statistik dataset
- **`evaluation/evaluate_all_models.py`** - Evaluasi performa semua model

## 🛠️ Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/RoboGaje/train-eval-model.git
cd train-eval-model
```

### 2. Setup Environment
```bash
# Buat conda environment
conda create -n robotika-uas python=3.11
conda activate robotika-uas

# Install dependencies
pip install -r requirements.txt
```

### 3. Verifikasi Dataset
```bash
python check_dataset.py
```

## 🎯 Penggunaan

### 🚀 Quick Start - Inference

**Masuk ke direktori inference:**
```bash
cd inference
```

**Video processing dengan YOLO + FaceNet:**
```bash
python facenet_yolo_inference.py \
  --yolo-model ../models/YOLO12n/weights/best.pt \
  --facenet-model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode video \
  --input ../WIN_20250612_17_21_33_Pro.mp4 \
  --output ../result.mp4 \
  --use-tensorrt
```

**Real-time webcam tracking:**
```bash
python realtime_tracking.py --runtime tensorrt --show-fps
```

### 1. Training Model (Opsional - Model sudah tersedia)

#### YOLO Training
```bash
# Training semua model
python train_yolo_models.py

# Training model spesifik
python train_yolo_models.py --models yolov12n yolov12s
```

#### FaceNet Training
```bash
# Training FaceNet (akan membuat folder dengan timestamp)
python train_facenet.py --dataset facenet_dataset --epochs 50

# Lihat history training
python train_facenet.py --list-trainings
```

### 2. Demo Tracking (Tanpa Display)

**Masuk ke direktori inference:**
```bash
cd inference
```

#### Demo dengan Test Images
```bash
python demo_tracking.py --mode images
```
**Output:** Folder `../demo_output_images/` berisi hasil deteksi

#### Demo Perbandingan PyTorch vs TensorRT
```bash
python demo_tracking.py --mode compare
```

#### Demo dengan Video Input
```bash
# Menggunakan TensorRT (lebih cepat)
python demo_tracking.py \
  --mode video \
  --input ../WIN_20250612_17_21_33_Pro.mp4 \
  --output ../demo_result.mp4 \
  --runtime tensorrt
```

### 3. Real-time Tracking dengan Webcam

**Masuk ke direktori inference:**
```bash
cd inference
```

#### Menggunakan TensorRT (Tercepat)
```bash
python realtime_tracking.py --runtime tensorrt --show-fps
```

#### Dengan Video Recording
```bash
python realtime_tracking.py \
  --runtime tensorrt \
  --save-video ../realtime_output.mp4 \
  --show-fps
```

**Kontrol Keyboard:**
- `SPACE` - Pause/Resume
- `Q` atau `ESC` - Quit
- `S` - Save current frame

### 4. FaceNet + YOLO Inference

**Masuk ke direktori inference:**
```bash
cd inference
```

#### Image Processing
```bash
python facenet_yolo_inference.py \
  --yolo-model ../models/YOLO12n/weights/best.pt \
  --facenet-model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode image \
  --input ../test/images/sample.jpg \
  --output ../result.jpg \
  --show
```

#### Video Processing dengan Frame Interval
```bash
python facenet_yolo_inference.py \
  --yolo-model ../models/YOLO12n/weights/best.pt \
  --facenet-model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode video \
  --input ../WIN_20250612_17_21_33_Pro.mp4 \
  --output ../result.mp4 \
  --use-tensorrt \
  --frame-interval 3 \
  --yolo-conf 0.5 \
  --facenet-conf 0.7
```

### 5. Evaluasi Model

```bash
cd evaluation/
python evaluate_all_models.py
```

## 📊 Hasil Benchmark

### Runtime Performance (YOLOv12n)
| Engine | Avg Time | FPS | Speedup |
|--------|----------|-----|---------|
| **TensorRT** | **5.2ms** | **190.9** | **7.1x** |
| PyTorch | 17.3ms | 57.7 | 1.0x |
| OpenVINO | 38.0ms | 26.3 | 0.5x |
| ONNX | 86.9ms | 11.5 | 0.2x |
| NCNN | 98.8ms | 10.1 | 0.2x |

### Model Accuracy
#### YOLO (mAP@0.5)
| Model | mAP@0.5 | mAP@0.5:0.95 | Model Size |
|-------|---------|--------------|------------|
| YOLOv12n | 0.945 | 0.830 | 5.3MB |
| YOLOv12s | ~0.95 | ~0.84 | 54MB |
| YOLOv12m | ~0.96 | ~0.85 | 39MB |
| YOLOv12l | ~0.97 | ~0.86 | 51MB |
| YOLOv12x | ~0.98 | ~0.87 | 114MB |

#### FaceNet
| Metric | Value |
|--------|-------|
| Validation Accuracy | 98.50% |
| Test Accuracy | 95.70% |
| Classes | 5 (dimas, fabian, sendy, syahrul, unknown) |

## ⚡ TensorRT Support

Untuk performa maksimal, gunakan TensorRT engine:

1. **Automatic Detection**: Script akan otomatis mencari file `.engine`
2. **Manual Path**: Langsung gunakan path ke file `.engine`

```bash
# Automatic (recommended)
--yolo-model ../models/YOLO12n/weights/best.pt --use-tensorrt

# Manual
--yolo-model ../models/YOLO12n/weights/best.engine
```

## 🎮 Mode Demo yang Tersedia

### 1. Images Mode
```bash
python demo_tracking.py --mode images
```
- Memproses 5 gambar test pertama
- Output: `demo_output_images/result_XX_filename.jpg`

### 2. Video Mode  
```bash
python demo_tracking.py --mode video --input video.mp4 --runtime tensorrt
```
- Memproses video dengan runtime pilihan
- Output: `demo_output_tensorrt.mp4`

### 3. Compare Mode
```bash
python demo_tracking.py --mode compare
```
- Membandingkan PyTorch vs TensorRT pada 1 gambar
- Output: File perbandingan

### 4. Video-Compare Mode
```bash
python demo_tracking.py --mode video-compare --input video.mp4
```
- Membandingkan PyTorch vs TensorRT pada video
- Output: 3 video (pytorch, tensorrt, comparison)

## 🔧 Troubleshooting

### 1. CUDA Issues
```bash
# Cek CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Jika TensorRT error, gunakan PyTorch
python realtime_tracking.py --runtime pytorch
```

### 2. Camera Issues
```bash
# Test camera lain
python realtime_tracking.py --camera 1

# Cek available cameras
ls /dev/video*
```

### 3. Memory Issues
```bash
# Gunakan model lebih kecil
python realtime_tracking.py --model yolov12n

# Kurangi image size
python realtime_tracking.py --imgsz 416
```

### 4. Display Issues (Server/Headless)
```bash
# Gunakan demo mode (export ke file)
python demo_tracking.py --mode images

# Atau video mode
python demo_tracking.py --mode video --input video.mp4
```

## 📁 Struktur File Output

```
robotika-uas/
├── demo_output_images/          # Hasil demo images
│   ├── result_01_*.jpg
│   ├── result_02_*.jpg
│   └── ...
├── comparison_*.jpg             # Hasil perbandingan
├── result_pytorch.jpg           # Hasil PyTorch
├── result_tensorrt.jpg          # Hasil TensorRT
├── output_*.mp4                 # Video output
├── frame_*.jpg                  # Saved frames
└── evaluation/
    ├── model_evaluation_results.csv
    └── model_comparison_visualizations.png
```

## 🚀 Quick Start

1. **Demo cepat dengan gambar:**
```bash
conda activate robotika-uas
python demo_tracking.py --mode compare
```

2. **Real-time tracking tercepat:**
```bash
python realtime_tracking.py --runtime tensorrt --show-fps
```

3. **Proses video dengan TensorRT:**
```bash
python demo_tracking.py --mode video --input your_video.mp4 --runtime tensorrt
```

## 📞 Support

Jika mengalami masalah:
1. Pastikan environment `robotika-uas` aktif
2. Cek ketersediaan model di folder `models/`
3. Untuk server tanpa display, gunakan `demo_tracking.py`
4. Untuk real-time, gunakan `realtime_tracking.py`

---

**🎉 Happy Tracking!** 🎯 