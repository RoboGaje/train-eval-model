# 🚀 Inference Scripts

Direktori ini berisi script-script untuk melakukan inference menggunakan model YOLO dan FaceNet yang sudah dilatih.

## 📁 Struktur Direktori

```
inference/
├── README.md                    # Dokumentasi ini
├── facenet_yolo_inference.py    # YOLO + FaceNet inference (utama)
├── facenet_inference.py         # FaceNet inference murni
├── demo_tracking.py             # Demo tracking dengan output file
└── realtime_tracking.py         # Real-time tracking dengan webcam
```

## 🎯 Script Utama

### 1. **facenet_yolo_inference.py** - YOLO + FaceNet Inference

Script utama yang menggabungkan YOLO untuk face detection dan FaceNet untuk face recognition.

**Fitur:**
- ✅ Dual prediction: YOLO classification + FaceNet recognition
- ⚡ TensorRT support untuk performa maksimal
- 🎬 Support image dan video processing
- 🔧 Configurable frame intervals untuk video

**Usage:**

```bash
# Video processing dengan TensorRT
python facenet_yolo_inference.py \
  --yolo-model ../models/YOLO12n/weights/best.pt \
  --facenet-model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode video \
  --input ../WIN_20250612_17_21_33_Pro.mp4 \
  --output ../result.mp4 \
  --use-tensorrt \
  --frame-interval 1 \
  --yolo-conf 0.5 \
  --facenet-conf 0.7

# Image processing
python facenet_yolo_inference.py \
  --yolo-model ../models/YOLO12n/weights/best.pt \
  --facenet-model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode image \
  --input ../test/images/sample.jpg \
  --output ../result.jpg \
  --show
```

**Parameters:**
- `--yolo-model`: Path ke YOLO model (.pt atau .engine)
- `--facenet-model`: Path ke FaceNet model (.pth)
- `--mapping`: Path ke class mapping file (.pkl)
- `--mode`: Mode inference (image/video)
- `--input`: Input file path
- `--output`: Output file path
- `--use-tensorrt`: Gunakan TensorRT engine untuk YOLO
- `--frame-interval`: Interval frame processing (1=setiap frame, 3=setiap 3 frame)
- `--yolo-conf`: YOLO confidence threshold (default: 0.5)
- `--facenet-conf`: FaceNet confidence threshold (default: 0.7)
- `--show`: Tampilkan hasil

### 2. **facenet_inference.py** - FaceNet Pure Recognition

Script untuk inference FaceNet murni dengan MTCNN face detection.

**Fitur:**
- 🎯 Pure FaceNet recognition
- 📷 MTCNN face detection
- 🎬 Support image, video, dan real-time
- 📸 Save frame functionality

**Usage:**

```bash
# Video processing
python facenet_inference.py \
  --model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode video \
  --input ../WIN_20250612_17_21_33_Pro.mp4 \
  --output ../facenet_result.mp4

# Real-time webcam
python facenet_inference.py \
  --model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode realtime \
  --camera 0
```

### 3. **demo_tracking.py** - Demo dengan Output File

Script demo untuk testing model dengan berbagai mode tanpa display (headless).

**Fitur:**
- 🖼️ Demo dengan test images
- 🎬 Demo dengan video
- ⚡ Perbandingan PyTorch vs TensorRT
- 📊 Performance benchmarking

**Usage:**

```bash
# Demo dengan test images
python demo_tracking.py --mode images

# Demo video dengan TensorRT
python demo_tracking.py \
  --mode video \
  --input ../WIN_20250612_17_21_33_Pro.mp4 \
  --output ../demo_result.mp4 \
  --runtime tensorrt

# Perbandingan PyTorch vs TensorRT
python demo_tracking.py --mode compare

# Perbandingan video
python demo_tracking.py \
  --mode video-compare \
  --input ../WIN_20250612_17_21_33_Pro.mp4
```

### 4. **realtime_tracking.py** - Real-time Webcam Tracking

Script untuk real-time object tracking menggunakan webcam.

**Fitur:**
- 📹 Real-time webcam processing
- ⚡ Multiple runtime support (PyTorch/TensorRT)
- 🎮 Interactive controls
- 💾 Video recording
- 📊 FPS monitoring

**Usage:**

```bash
# Real-time dengan PyTorch
python realtime_tracking.py --runtime pytorch

# Real-time dengan TensorRT (fastest)
python realtime_tracking.py --runtime tensorrt --show-fps

# Dengan video recording
python realtime_tracking.py \
  --runtime tensorrt \
  --save-video ../realtime_output.mp4 \
  --show-fps
```

**Controls:**
- `SPACE`: Pause/Resume
- `Q/ESC`: Quit
- `S`: Save current frame

## 🔧 Model Paths

Semua script menggunakan relative paths dari direktori `inference/`:

```
../models/
├── YOLO12n/weights/
│   ├── best.pt          # PyTorch model
│   └── best.engine      # TensorRT engine
├── YOLO12s/weights/best.pt
├── YOLOv12m/weights/best.pt
├── YOLOv12l/weights/best.pt
├── YOLOv12x/weights/best.pt
└── facenet_models/
    ├── latest/          # Symlink ke training terbaru
    │   ├── best_facenet.pth
    │   └── class_mapping.pkl
    └── training_YYYYMMDD_HHMMSS/
        ├── best_facenet.pth
        └── class_mapping.pkl
```

## ⚡ TensorRT Support

Untuk performa maksimal, gunakan TensorRT engine:

1. **Automatic**: Script akan otomatis mencari file `.engine` jika flag `--use-tensorrt` digunakan
2. **Manual**: Langsung gunakan path ke file `.engine`

```bash
# Automatic TensorRT detection
--yolo-model ../models/YOLO12n/weights/best.pt --use-tensorrt

# Manual TensorRT path
--yolo-model ../models/YOLO12n/weights/best.engine
```

## 📊 Performance Tips

1. **Frame Interval**: Gunakan `--frame-interval 3` untuk video processing yang lebih cepat
2. **TensorRT**: Selalu gunakan TensorRT untuk performa terbaik
3. **Confidence Threshold**: Sesuaikan threshold untuk balance antara accuracy dan speed
4. **Image Size**: Gunakan `--imgsz 640` untuk balance antara accuracy dan speed

## 🎯 Class Names

Model mendukung 5 classes:
- `dimas` - Warna: Biru
- `fabian` - Warna: Hijau  
- `people face` / `unknown` - Warna: Merah
- `sendy` - Warna: Cyan
- `syahrul` - Warna: Magenta

## 🚨 Troubleshooting

### Model tidak ditemukan
```bash
❌ Error: Model file not found
```
**Solusi**: Pastikan path model benar dan file ada

### TensorRT engine tidak ditemukan
```bash
⚠️ TensorRT engine not found, using PyTorch
```
**Solusi**: File `.engine` akan dicari otomatis, atau buat dengan:
```bash
yolo export model=../models/YOLO12n/weights/best.pt format=engine
```

### GPU memory error
```bash
❌ CUDA out of memory
```
**Solusi**: 
- Kurangi batch size
- Gunakan model yang lebih kecil (YOLOv12n)
- Gunakan `--frame-interval` yang lebih besar

### Webcam tidak terbuka
```bash
❌ Error: Cannot open camera 0
```
**Solusi**:
- Coba camera index lain: `--camera 1`
- Pastikan webcam tidak digunakan aplikasi lain
- Check permission webcam

## 📝 Examples

### Quick Start - Video Processing
```bash
cd inference
python facenet_yolo_inference.py \
  --yolo-model ../models/YOLO12n/weights/best.pt \
  --facenet-model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode video \
  --input ../WIN_20250612_17_21_33_Pro.mp4 \
  --output ../result.mp4 \
  --use-tensorrt
```

### Quick Start - Real-time Webcam
```bash
cd inference
python realtime_tracking.py --runtime tensorrt --show-fps
```

### Quick Start - Demo
```bash
cd inference
python demo_tracking.py --mode images
``` 