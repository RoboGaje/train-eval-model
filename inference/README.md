# 🚀 Inference Scripts

Direktori ini berisi script-script untuk melakukan inference menggunakan model YOLO dan FaceNet yang sudah dilatih.

## 📁 Struktur Direktori

```
inference/
├── README.md                      # Dokumentasi ini
├── facenet_yolo_inference.py      # YOLO + FaceNet inference (utama)
├── facenet_inference.py           # FaceNet inference murni
├── demo_tracking.py               # Demo tracking dengan output file
├── realtime_tracking.py           # Real-time tracking dengan webcam
└── body_detection_inference.py    # Body detection dengan YOLO pre-trained
```

## 🎯 YOLO + FaceNet Face Recognition System

### **Overview**
Sistem face recognition hybrid yang menggabungkan:
- **YOLO**: Face detection (mendeteksi lokasi wajah)
- **FaceNet**: Face recognition (mengidentifikasi identitas wajah)

### **Fitur Utama**

#### 🔍 **Dual Prediction System**
- Menampilkan prediksi YOLO dan FaceNet secara bersamaan
- YOLO: Detection + classification wajah
- FaceNet: Pure face recognition dengan akurasi tinggi (98.8%)

#### ⚡ **TensorRT Support**
- Support TensorRT engine untuk YOLO (3-5x lebih cepat)
- Automatic fallback ke model regular jika engine tidak tersedia
- Optimized untuk GPU inference

#### 🎛️ **Configurable Frame Processing**
- `--frame-interval 1`: Process setiap frame (smooth, lambat)
- `--frame-interval 3`: Balance optimal (default)
- `--frame-interval 5`: Cepat, kualitas acceptable
- Smart interpolation untuk frame yang tidak diproses

#### 🏷️ **Consistent Labeling**
- Automatic mapping "people face" → "unknown"
- Konsistensi dengan FaceNet class names
- Support custom class mapping

### **Performance Comparison**

#### **Regular YOLO vs TensorRT**
| Mode | Time | FPS | Improvement |
|------|------|-----|-------------|
| Regular PT | 26.9s | 25.7 FPS | Baseline |
| TensorRT | 23.6s | 29.4 FPS | **+14% faster** |

#### **Frame Interval Impact**
| Interval | Time | FPS | Quality |
|----------|------|-----|---------|
| 1 (every frame) | 45.5s | 15.2 FPS | ⭐⭐⭐⭐⭐ |
| 3 (default) | 26.9s | 25.7 FPS | ⭐⭐⭐⭐ |
| 5 (fast) | 22.7s | 30.5 FPS | ⭐⭐⭐ |

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
# Basic Image Processing
python facenet_yolo_inference.py \
  --yolo-model ../models/YOLO12n/weights/best.pt \
  --facenet-model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode image \
  --input ../test/images/sample.jpg \
  --output ../result.jpg \
  --show

# Video Processing with TensorRT
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

# High Quality Processing
python facenet_yolo_inference.py \
  --yolo-model ../models/YOLO12n/weights/best.pt \
  --facenet-model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode video \
  --input ../video.mp4 \
  --output ../result_hq.mp4 \
  --frame-interval 1 \
  --yolo-conf 0.3 \
  --facenet-conf 0.8
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yolo-model` | Required | Path ke YOLO model (.pt atau .engine) |
| `--facenet-model` | Required | Path ke FaceNet model (.pth) |
| `--mapping` | Auto-detect | Path ke class mapping file |
| `--mode` | image | Mode: image atau video |
| `--input` | Required | Input file path |
| `--output` | Optional | Output file path |
| `--yolo-conf` | 0.5 | YOLO confidence threshold |
| `--facenet-conf` | 0.7 | FaceNet confidence threshold |
| `--frame-interval` | 3 | Frame processing interval |
| `--use-tensorrt` | False | Enable TensorRT acceleration |
| `--device` | Auto | Device: cuda atau cpu |
| `--show` | False | Display results |

**Output Format:**

#### **Bounding Box Labels**
- **Top (Yellow)**: `YOLO: [class] ([confidence])`
- **Bottom (Green/Red)**: `FaceNet: [identity] ([confidence])`
- **Color**: Green untuk known faces, Red untuk unknown

#### **Console Output**
```
✅ Detected 1 faces
   Face 1:
     YOLO: sendy (conf: 0.919)
     FaceNet: syahrul (conf: 0.976)
```

### 2. **body_detection_inference.py** - Body Detection dengan YOLO Pre-trained

Script untuk body detection menggunakan YOLO12n pre-trained model dengan COCO dataset.

**Fitur:**
- 🎯 **Person Detection**: Deteksi hanya class 'person' menggunakan post-processing filtering
- ⚡ **COCO Pre-trained**: Menggunakan model yang sudah dilatih dengan 80 classes
- 🎬 **Video Support**: Support untuk image dan video processing
- 📊 **Performance Info**: Menampilkan inference time dan FPS

**Approach:**

#### **Post-processing Filtering**
- Load YOLO pre-trained model normal (80 classes)
- Lakukan inference dengan parameter `classes=[0]` untuk optimasi
- Filter hasil untuk hanya ambil class 'person' (index 0)
- **Pros**: Mudah, stabil, tidak perlu modifikasi model
- **Performance**: ~18ms inference time, 38+ FPS untuk video

**Usage:**

```bash
# Download YOLO pre-trained model dulu
# pip install ultralytics
# python -c "from ultralytics import YOLO; YOLO('yolo12n.pt')"

# Basic Body Detection
python body_detection_inference.py \
  --model yolo12n.pt \
  --mode image \
  --input ../test/images/sample.jpg \
  --output ../body_result.jpg \
  --show

# Video Processing
python body_detection_inference.py \
  --model yolo12n.pt \
  --mode video \
  --input ../WIN_20250612_17_21_33_Pro.mp4 \
  --output ../body_result.mp4
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | yolo12n.pt | Path ke YOLO pre-trained model |
| `--mode` | image | Mode: image atau video |
| `--input` | Required | Input file path |
| `--output` | Optional | Output file path |
| `--confidence` | 0.5 | Confidence threshold untuk person detection |
| `--show` | False | Display results |
| `--device` | Auto | Device: cuda atau cpu |

### 3. **facenet_inference.py** - FaceNet Pure Recognition

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

### 4. **demo_tracking.py** - Demo dengan Output File

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

### 5. **realtime_tracking.py** - Real-time Webcam Tracking

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

## 🔧 Technical Details

### **Architecture**
```
Input Image/Video
    ↓
YOLO Detection (Bounding Boxes)
    ↓
Face Cropping
    ↓
FaceNet Recognition (Identity)
    ↓
Dual Label Display
```

### **Class Mapping**
- YOLO classes: `['dimas', 'fabian', 'people face', 'sendy', 'syahrul']`
- Mapped to: `['dimas', 'fabian', 'unknown', 'sendy', 'syahrul']`
- FaceNet classes: `['dimas', 'fabian', 'sendy', 'syahrul', 'unknown']`

### **COCO Classes (Body Detection)**
- Total: 80 classes
- Person class: Index 0
- Full list: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, ...

### **Model Requirements**
- **YOLO**: YOLOv12n trained model (.pt) atau TensorRT engine (.engine)
- **FaceNet**: Custom trained model dengan InceptionResnetV1 backbone
- **YOLO Pre-trained**: COCO pre-trained models untuk body detection
- **GPU**: CUDA-capable GPU recommended untuk performa optimal

## 🔧 Model Paths

Semua script menggunakan relative paths dari direktori `inference/`:

```
../models/
├── YOLO12n/weights/
│   ├── best.pt          # PyTorch model (custom trained)
│   └── best.engine      # TensorRT engine
├── YOLO12s/weights/best.pt
├── YOLOv12m/weights/best.pt
├── YOLOv12l/weights/best.pt
├── YOLOv12x/weights/best.pt
├── pretrained/          # COCO pre-trained models
│   ├── yolo12n.pt       # Body detection
│   ├── yolo12s.pt
│   └── yolo12m.pt
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
5. **Body Detection**: Gunakan filtering approach untuk stability dan performa optimal

## 🎯 Class Names

### **Face Recognition Model**
Model mendukung 5 classes:
- `dimas` - Warna: Biru
- `fabian` - Warna: Hijau  
- `people face` / `unknown` - Warna: Merah
- `sendy` - Warna: Cyan
- `syahrul` - Warna: Magenta

### **Body Detection Model (COCO)**
Model mendukung 80 classes, fokus pada:
- `person` (index 0) - Warna: Hijau
- Filtering untuk hanya deteksi bodies

## 🎯 Use Cases

### **Face Recognition System**
1. **Security Systems**: Real-time face recognition dengan dual verification
2. **Attendance Systems**: Akurat identification dengan confidence scoring
3. **Video Analytics**: Batch processing video dengan optimized performance
4. **Research**: Comparison antara detection dan recognition models

### **Body Detection System**
1. **People Counting**: Hitung jumlah orang di area tertentu
2. **Crowd Analysis**: Analisis kepadatan dan pergerakan crowd
3. **Security Monitoring**: Deteksi kehadiran orang di area restricted
4. **Retail Analytics**: Customer flow analysis di toko

## 🚨 Troubleshooting

### **TensorRT Issues**
- Pastikan TensorRT engine compatible dengan GPU architecture
- Fallback otomatis ke regular model jika engine gagal load

### **Performance Optimization**
- Gunakan `--use-tensorrt` untuk YOLO acceleration
- Adjust `--frame-interval` sesuai kebutuhan speed vs quality
- Lower confidence thresholds untuk detection lebih sensitif

### **Memory Issues**
- Reduce `--frame-interval` untuk mengurangi memory usage
- Use CPU mode dengan `--device cpu` jika GPU memory terbatas

### **Body Detection Issues**
- **Model tidak ditemukan**: Jalankan `download_yolo_pretrained.py` dulu
- **Akurasi rendah**: Turunkan confidence threshold atau gunakan model yang lebih besar

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

### Quick Start - Face Recognition
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

### Quick Start - Body Detection
```bash
cd inference

# Download model dulu (otomatis via ultralytics)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo12n.pt')"

# Body detection
python body_detection_inference.py \
  --model yolo12n.pt \
  --mode video \
  --input ../WIN_20250612_17_21_33_Pro.mp4 \
  --output ../body_result.mp4
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

## 🔄 Workflow Recommendations

### **1. Setup Models**
```bash
# Download YOLO pre-trained untuk body detection (otomatis via ultralytics)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo12n.pt')"

# Pastikan custom trained models ada
ls ../models/YOLO12n/weights/best.pt
ls ../models/facenet_models/latest/
```

### **2. Production Inference**
```bash
# Face recognition (custom trained)
python facenet_yolo_inference.py \
  --yolo-model ../models/YOLO12n/weights/best.pt \
  --facenet-model ../models/facenet_models/latest/best_facenet.pth \
  --mode video \
  --input ../video.mp4 \
  --use-tensorrt

# Body detection (pre-trained)
python body_detection_inference.py \
  --model yolo12n.pt \
  --mode video \
  --input ../video.mp4
```