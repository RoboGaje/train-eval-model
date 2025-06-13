# 🚀 Inference Scripts

Direktori ini berisi script-script untuk melakukan inference menggunakan model YOLO dan FaceNet.

## 📁 Struktur Direktori

```
inference/
├── README.md                      # Dokumentasi ini
├── facenet_yolo_inference.py      # YOLO + FaceNet inference (utama)
├── facenet_inference.py           # FaceNet inference murni
├── demo_tracking.py               # Demo tracking dengan output file
├── realtime_tracking.py          # Real-time tracking dengan webcam
├── body_detection_inference.py    # Body detection dengan YOLO pre-trained
└── face_body_detection_inference.py # Face + Body detection dengan analisis kepadatan
```

## 🎯 Script Utama

### 1. **face_body_detection_inference.py** - Face + Body Detection dengan Analisis Kepadatan

Script yang menggabungkan deteksi wajah dan body dengan analisis kepadatan crowd.

#### **Overview**
Sistem dual detection yang menggabungkan:
- **Face Detection**: YOLO fine-tuned model untuk deteksi wajah
- **Body Detection**: YOLO pre-trained model (COCO) untuk deteksi body
- **Density Analysis**: Analisis kepadatan dan statistik crowd

#### **Fitur Utama**
- 🎯 **Dual Detection**: Deteksi wajah dan body secara bersamaan
- 📊 **Density Analysis**: Analisis kepadatan area dan crowd level
- 📈 **Statistics**: Statistik lengkap per frame dan keseluruhan
- ⚡ **TensorRT Support**: Support untuk kedua model
- 🎨 **Visual Output**: Bounding box berbeda warna untuk face (biru) dan body (hijau)
- 💾 **Export Stats**: Export statistik ke file JSON

#### **Crowd Level Classification**
- **Empty**: 0 orang
- **Low**: 1-2 orang
- **Medium**: 3-5 orang
- **High**: 6-10 orang
- **Very High**: >10 orang

**Usage:**

```bash
# Basic Image Processing
python face_body_detection_inference.py \
  --face-model ../models/YOLO12n_finetuned/weights/best.pt \
  --body-model ../models/YOLO12n_pretrained/yolo12n.pt \
  --mode image \
  --input ../test/images/sample.jpg \
  --output ../face_body_result.jpg \
  --show

# Video Processing dengan TensorRT dan Statistics
python face_body_detection_inference.py \
  --face-model ../models/YOLO12n_finetuned/weights/best.engine \
  --body-model ../models/YOLO12n_pretrained/yolo12n.engine \
  --mode video \
  --input ../video.mp4 \
  --output ../face_body_result.mp4 \
  --use-tensorrt \
  --save-stats \
  --face-conf 0.5 \
  --body-conf 0.5
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--face-model` | Required | Path ke YOLO face detection model |
| `--body-model` | yolo12n.pt | Path ke YOLO body detection model |
| `--mode` | image | Mode: image atau video |
| `--input` | Required | Input file path |
| `--output` | Optional | Output file path |
| `--face-conf` | 0.5 | Face detection confidence threshold |
| `--body-conf` | 0.5 | Body detection confidence threshold |
| `--show` | False | Display results |
| `--device` | Auto | Device: cuda atau cpu |
| `--use-tensorrt` | False | Enable TensorRT acceleration |
| `--save-stats` | False | Save statistics to JSON file |

### 2. **facenet_yolo_inference.py** - YOLO + FaceNet Inference

Script utama yang menggabungkan YOLO untuk face detection dan FaceNet untuk face recognition.

#### **Overview**
Sistem face recognition hybrid yang menggabungkan:
- **YOLO**: Face detection (mendeteksi lokasi wajah)
- **FaceNet**: Face recognition (mengidentifikasi identitas wajah)

#### **Fitur Utama**
- ✅ Dual prediction: YOLO classification + FaceNet recognition
- ⚡ TensorRT support untuk performa maksimal
- 🎬 Support image dan video processing
- 🔧 Configurable frame intervals untuk video

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

#### **Technical Details**

##### **Architecture**
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

##### **Class Mapping**
- YOLO classes: `['dimas', 'fabian', 'people face', 'sendy', 'syahrul']`
- Mapped to: `['dimas', 'fabian', 'unknown', 'sendy', 'syahrul']`
- FaceNet classes: `['dimas', 'fabian', 'sendy', 'syahrul', 'unknown']`

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
  --input ../video.mp4 \
  --output ../result.mp4 \
  --use-tensorrt \
  --frame-interval 3 \
  --yolo-conf 0.5 \
  --facenet-conf 0.7
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

### 3. **body_detection_inference.py** - Body Detection dengan YOLO Pre-trained

Script untuk body detection menggunakan YOLO12n pre-trained model dengan COCO dataset.

**Fitur:**
- 🎯 **Person Detection**: Deteksi hanya class 'person' menggunakan post-processing filtering
- ⚡ **COCO Pre-trained**: Menggunakan model yang sudah dilatih dengan 80 classes
- 🎬 **Video Support**: Support untuk image dan video processing
- 📊 **Performance Info**: Menampilkan inference time dan FPS
- ⚡ **TensorRT Support**: Support TensorRT engine untuk optimasi performa (lebih cepat)

**Approach:**

#### **Post-processing Filtering**
- Load YOLO pre-trained model normal (80 classes)
- Lakukan inference dengan parameter `classes=[0]` untuk optimasi
- Filter hasil untuk hanya ambil class 'person' (index 0)
- **Pros**: Mudah, stabil, tidak perlu modifikasi model
- **Performance**: ~18ms inference time, 38+ FPS untuk video
- **TensorRT**: ~5ms inference time, 120+ FPS dengan TensorRT engine

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

# Video Processing dengan TensorRT
python body_detection_inference.py \
  --model yolo12n.engine \
  --mode video \
  --input ../video.mp4 \
  --output ../body_result.mp4 \
  --use-tensorrt \
  --confidence 0.5
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | yolo12n.pt | Path ke YOLO pre-trained model (.pt atau .engine) |
| `--mode` | image | Mode: image atau video |
| `--input` | Required | Input file path |
| `--output` | Optional | Output file path |
| `--confidence` | 0.5 | Confidence threshold untuk person detection |
| `--show` | False | Display results |
| `--device` | Auto | Device: cuda atau cpu |
| `--use-tensorrt` | False | Enable TensorRT acceleration |

### 4. **facenet_inference.py** - FaceNet Pure Recognition

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
  --input ../video.mp4 \
  --output ../facenet_result.mp4

# Real-time webcam
python facenet_inference.py \
  --model ../models/facenet_models/latest/best_facenet.pth \
  --mapping ../models/facenet_models/latest/class_mapping.pkl \
  --mode realtime \
  --camera 0
```

### 5. **demo_tracking.py** - Demo dengan Output File

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
  --input ../video.mp4 \
  --output ../demo_result.mp4 \
  --runtime tensorrt
```

### 6. **realtime_tracking.py** - Real-time Webcam Tracking

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
├── YOLO12n_finetuned/weights/
│   ├── best.pt          # PyTorch model (face detection)
│   └── best.engine      # TensorRT engine (face detection)
├── YOLO12n_pretrained/
│   ├── yolo12n.pt       # PyTorch model (body detection)
│   └── yolo12n.engine   # TensorRT engine (body detection)
├── YOLO12s_finetuned/weights/best.pt
├── YOLOv12m_finetuned/weights/best.pt
├── YOLOv12l_finetuned/weights/best.pt
├── YOLOv12x_finetuned/weights/best.pt
└── facenet_models/
    ├── README.md            # Download instructions
    ├── best_facenet.pth     # FaceNet model (download required)
    └── class_mapping.pkl    # Class mapping file
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
