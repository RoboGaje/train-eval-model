# YOLO + FaceNet Face Recognition System

## 🎯 Overview
Sistem face recognition hybrid yang menggabungkan:
- **YOLO**: Face detection (mendeteksi lokasi wajah)
- **FaceNet**: Face recognition (mengidentifikasi identitas wajah)

## ✨ Fitur Utama

### 🔍 **Dual Prediction System**
- Menampilkan prediksi YOLO dan FaceNet secara bersamaan
- YOLO: Detection + classification wajah
- FaceNet: Pure face recognition dengan akurasi tinggi (98.8%)

### ⚡ **TensorRT Support**
- Support TensorRT engine untuk YOLO (3-5x lebih cepat)
- Automatic fallback ke model regular jika engine tidak tersedia
- Optimized untuk GPU inference

### 🎛️ **Configurable Frame Processing**
- `--frame-interval 1`: Process setiap frame (smooth, lambat)
- `--frame-interval 3`: Balance optimal (default)
- `--frame-interval 5`: Cepat, kualitas acceptable
- Smart interpolation untuk frame yang tidak diproses

### 🏷️ **Consistent Labeling**
- Automatic mapping "people face" → "unknown"
- Konsistensi dengan FaceNet class names
- Support custom class mapping

## 📊 Performance Comparison

### **Regular YOLO vs TensorRT**
| Mode | Time | FPS | Improvement |
|------|------|-----|-------------|
| Regular PT | 26.9s | 25.7 FPS | Baseline |
| TensorRT | 23.6s | 29.4 FPS | **+14% faster** |

### **Frame Interval Impact**
| Interval | Time | FPS | Quality |
|----------|------|-----|---------|
| 1 (every frame) | 45.5s | 15.2 FPS | ⭐⭐⭐⭐⭐ |
| 3 (default) | 26.9s | 25.7 FPS | ⭐⭐⭐⭐ |
| 5 (fast) | 22.7s | 30.5 FPS | ⭐⭐⭐ |

## 🚀 Usage

### **Basic Image Processing**
```bash
python facenet_yolo_inference.py \
  --yolo-model models/YOLO12n/weights/best.pt \
  --facenet-model models/facenet_models/best_facenet.pth \
  --mapping models/facenet_models/class_mapping.pkl \
  --mode image \
  --input test_image.jpg \
  --output result.jpg
```

### **Video Processing with TensorRT**
```bash
python facenet_yolo_inference.py \
  --yolo-model models/YOLO12n/weights/best.pt \
  --facenet-model models/facenet_models/best_facenet.pth \
  --mapping models/facenet_models/class_mapping.pkl \
  --mode video \
  --input video.mp4 \
  --output result.mp4 \
  --use-tensorrt \
  --frame-interval 3 \
  --yolo-conf 0.5 \
  --facenet-conf 0.7
```

### **High Quality Processing**
```bash
python facenet_yolo_inference.py \
  --yolo-model models/YOLO12n/weights/best.pt \
  --facenet-model models/facenet_models/best_facenet.pth \
  --mapping models/facenet_models/class_mapping.pkl \
  --mode video \
  --input video.mp4 \
  --output result_hq.mp4 \
  --frame-interval 1 \
  --yolo-conf 0.3 \
  --facenet-conf 0.8
```

## 📋 Parameters

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

## 🎨 Output Format

### **Bounding Box Labels**
- **Top (Yellow)**: `YOLO: [class] ([confidence])`
- **Bottom (Green/Red)**: `FaceNet: [identity] ([confidence])`
- **Color**: Green untuk known faces, Red untuk unknown

### **Console Output**
```
✅ Detected 1 faces
   Face 1:
     YOLO: sendy (conf: 0.919)
     FaceNet: syahrul (conf: 0.976)
```

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

### **Model Requirements**
- **YOLO**: YOLOv12n trained model (.pt) atau TensorRT engine (.engine)
- **FaceNet**: Custom trained model dengan InceptionResnetV1 backbone
- **GPU**: CUDA-capable GPU recommended untuk performa optimal

## 🎯 Use Cases

1. **Security Systems**: Real-time face recognition dengan dual verification
2. **Attendance Systems**: Akurat identification dengan confidence scoring
3. **Video Analytics**: Batch processing video dengan optimized performance
4. **Research**: Comparison antara detection dan recognition models

## 🔍 Troubleshooting

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