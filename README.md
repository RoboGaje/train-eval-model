# 🚀 YOLO Model Training & Real-time Tracking

Repository ini berisi implementasi lengkap untuk fine-tuning model YOLO dan real-time object tracking dengan berbagai runtime engine.

## 📋 Daftar Isi

- [Dataset](#-dataset)
- [Model yang Tersedia](#-model-yang-tersedia)
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

| Model | Ukuran File | Runtime | Lokasi |
|-------|-------------|---------|--------|
| YOLOv12n | 5.3MB | PyTorch | `models/YOLO12n/weights/best.pt` |
| YOLOv12n | 19MB | TensorRT | `models/YOLO12n/weights/best.engine` |
| YOLOv12s | 54MB | PyTorch | `models/YOLO12s/weights/best.pt` |
| YOLOv12m | ~39MB | PyTorch | `models/YOLOv12m/weights/best.pt` |
| YOLOv12l | ~51MB | PyTorch | `models/YOLOv12l/weights/best.pt` |
| YOLOv12x | ~114MB | PyTorch | `models/YOLOv12x/weights/best.pt` |

## 📜 Script yang Tersedia

### 1. Training Script
- **File:** `train_yolo_models.py`
- **Fungsi:** Fine-tuning semua model YOLO (n, s, m, l, x)
- **Fitur:** Batch size otomatis berdasarkan ukuran model

### 2. Dataset Checker
- **File:** `check_dataset.py`
- **Fungsi:** Verifikasi struktur dan statistik dataset

### 3. Model Evaluation
- **File:** `evaluation/evaluate_all_models.py`
- **Fungsi:** Evaluasi performa semua model pada validation set

### 4. Demo Tracking (Headless)
- **File:** `demo_tracking.py`
- **Fungsi:** Demo tracking dengan export ke file (tanpa display)

### 5. Real-time Tracking
- **File:** `realtime_tracking.py`
- **Fungsi:** Real-time tracking menggunakan webcam

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

### 1. Training Model (Opsional - Model sudah tersedia)

```bash
# Training semua model
python train_yolo_models.py

# Training model spesifik
python train_yolo_models.py --models yolov12n yolov12s
```

### 2. Demo Tracking (Tanpa Display)

#### Demo dengan Test Images
```bash
python demo_tracking.py --mode images
```
**Output:** Folder `demo_output_images/` berisi hasil deteksi

#### Demo Perbandingan PyTorch vs TensorRT
```bash
python demo_tracking.py --mode compare
```
**Output:** 
- `comparison_pytorch_vs_tensorrt.jpg` - Perbandingan side-by-side
- `result_pytorch.jpg` - Hasil PyTorch
- `result_tensorrt.jpg` - Hasil TensorRT

#### Demo dengan Video Input
```bash
# Menggunakan PyTorch
python demo_tracking.py --mode video --input input_video.mp4 --runtime pytorch

# Menggunakan TensorRT (lebih cepat)
python demo_tracking.py --mode video --input input_video.mp4 --runtime tensorrt --output output_video.mp4
```

#### Demo Perbandingan Video
```bash
python demo_tracking.py --mode video-compare --input input_video.mp4
```
**Output:**
- `output_pytorch.mp4` - Video hasil PyTorch
- `output_tensorrt.mp4` - Video hasil TensorRT  
- `output_comparison.mp4` - Video perbandingan side-by-side

### 3. Real-time Tracking dengan Webcam

#### Menggunakan PyTorch (Default)
```bash
python realtime_tracking.py --show-fps
```

#### Menggunakan TensorRT (Tercepat)
```bash
python realtime_tracking.py --runtime tensorrt --show-fps
```

#### Dengan Video Recording
```bash
python realtime_tracking.py --runtime tensorrt --save-video output.mp4 --show-fps
```

#### Opsi Lengkap
```bash
python realtime_tracking.py \
    --model yolov12n \
    --runtime tensorrt \
    --conf 0.5 \
    --iou 0.45 \
    --camera 0 \
    --imgsz 640 \
    --show-fps \
    --save-video tracking_output.mp4
```

**Kontrol Keyboard:**
- `SPACE` - Pause/Resume
- `Q` atau `ESC` - Quit
- `S` - Save current frame

### 4. Evaluasi Model

```bash
cd evaluation/
python evaluate_all_models.py
```
**Output:**
- `model_evaluation_results.csv` - Hasil evaluasi
- `model_comparison_visualizations.png` - Visualisasi perbandingan

## 📊 Hasil Benchmark

### Runtime Performance (YOLOv12n)
| Engine | Avg Time | FPS | Speedup |
|--------|----------|-----|---------|
| **TensorRT** | **5.2ms** | **190.9** | **7.1x** |
| PyTorch | 17.3ms | 57.7 | 1.0x |
| OpenVINO | 38.0ms | 26.3 | 0.5x |
| ONNX | 86.9ms | 11.5 | 0.2x |
| NCNN | 98.8ms | 10.1 | 0.2x |

### Model Accuracy (mAP@0.5)
| Model | mAP@0.5 | mAP@0.5:0.95 | Model Size |
|-------|---------|--------------|------------|
| YOLOv12n | 0.945 | 0.830 | 5.3MB |
| YOLOv12s | ~0.95 | ~0.84 | 54MB |
| YOLOv12m | ~0.96 | ~0.85 | 39MB |
| YOLOv12l | ~0.97 | ~0.86 | 51MB |
| YOLOv12x | ~0.98 | ~0.87 | 114MB |

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