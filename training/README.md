# 🏋️ Training Scripts

Direktori ini berisi script-script untuk melakukan training model YOLO dan FaceNet.

## 📁 Struktur Direktori

```
training/
├── README.md                    # Dokumentasi ini
├── train_yolo_models.py         # Training semua model YOLO
├── train_facenet.py             # Training FaceNet dengan timestamp folders
└── preprocess_for_facenet.py    # Preprocessing dataset untuk FaceNet
```

## 🎯 Script Training

### 1. **train_yolo_models.py** - Training Semua Model YOLO

Script untuk fine-tuning dan membandingkan performa semua model YOLO.

**Fitur:**
- ✅ Training 5 model YOLO: YOLOv12n, YOLOv12s, YOLOv12m, YOLOv12l, YOLOv12x
- 📊 Evaluasi dan perbandingan performa otomatis
- 📈 Visualisasi hasil perbandingan
- 💾 Export hasil ke CSV dan grafik

**Usage:**

```bash
# Training semua model dengan konfigurasi default
python train_yolo_models.py

# Model akan dilatih dengan batch size yang sudah dioptimalkan:
# - YOLOv12n: batch_size=100
# - YOLOv12s: batch_size=55
# - YOLOv12m: batch_size=30
# - YOLOv12l: batch_size=20
# - YOLOv12x: batch_size=13
```

**Output:**
- `../runs/detect/[MODEL_NAME]_epoch50_batch[SIZE]/` - Model hasil training
- `../model_comparison_results.csv` - Hasil perbandingan dalam CSV
- `../model_comparison_charts.png` - Grafik perbandingan performa

**Konfigurasi:**
- `EPOCHS = 50` - Jumlah epoch training
- `DATA_CONFIG = '../data.yaml'` - Path ke konfigurasi dataset

### 2. **train_facenet.py** - Training FaceNet dengan Timestamp

Script untuk fine-tuning FaceNet dengan sistem folder timestamp untuk menghindari overwrite.

**Fitur:**
- 🕐 Folder training dengan timestamp (training_YYYYMMDD_HHMMSS)
- 🔗 Symlink 'latest' ke training terbaru
- 📊 Evaluasi lengkap dengan confusion matrix
- 📈 Training curves dan visualisasi
- 🎯 Test pada gambar sample

**Usage:**

```bash
# Training dengan konfigurasi default
python train_facenet.py

# Training dengan parameter custom
python train_facenet.py \
  --dataset ../facenet_dataset \
  --output ../models/facenet_models \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001

# Lihat daftar training yang pernah dilakukan
python train_facenet.py --list-trainings

# Training dengan GPU spesifik
python train_facenet.py --gpu-id 0

# Force menggunakan CPU
python train_facenet.py --force-cpu
```

**Parameters:**
- `--dataset`: Path ke dataset FaceNet (default: ../facenet_dataset)
- `--output`: Output directory (default: ../models/facenet_models)
- `--epochs`: Jumlah epoch (default: 50)
- `--batch-size`: Batch size (default: 602)
- `--lr`: Learning rate (default: 0.001)
- `--test-images-dir`: Directory test images (default: ../test/images)
- `--num-test-images`: Jumlah test images (default: 10)

**Output Structure:**
```
../models/facenet_models/
├── latest/                      # Symlink ke training terbaru
│   ├── best_facenet.pth
│   └── class_mapping.pkl
└── training_20250101_120000/    # Folder dengan timestamp
    ├── best_facenet.pth         # Model terbaik
    ├── class_mapping.pkl        # Class mapping
    ├── training_curves.png      # Kurva training
    ├── confusion_matrix_test.png
    ├── class_accuracy_test.png
    ├── test_results/            # Hasil test pada gambar
    └── final_evaluation_summary.txt
```

### 3. **preprocess_for_facenet.py** - Preprocessing Dataset

Script untuk mengkonversi dataset YOLO ke format yang sesuai untuk training FaceNet.

**Fitur:**
- 🔄 Konversi YOLO bounding boxes ke cropped faces
- 📏 Resize otomatis ke 160x160 (standar FaceNet)
- 🏷️ Mapping "people face" → "unknown"
- 📊 Statistik dataset otomatis

**Usage:**

```bash
# Preprocessing dengan konfigurasi default
python preprocess_for_facenet.py

# Preprocessing dengan parameter custom
python preprocess_for_facenet.py \
  --data ../data.yaml \
  --output ../facenet_dataset \
  --padding 0.3
```

**Parameters:**
- `--data`: Path ke data.yaml (default: ../data.yaml)
- `--output`: Output directory (default: ../facenet_dataset)
- `--padding`: Padding untuk crop (default: 0.2)

**Output Structure:**
```
../facenet_dataset/
├── train/
│   ├── dimas/
│   ├── fabian/
│   ├── sendy/
│   ├── syahrul/
│   └── unknown/
├── val/
│   └── [same structure]
├── test/
│   └── [same structure]
└── dataset_summary.txt
```

## 🔄 Workflow Training Lengkap

### 1. Preprocessing Dataset untuk FaceNet
```bash
cd training
python preprocess_for_facenet.py
```

### 2. Training FaceNet
```bash
python train_facenet.py --epochs 50
```

### 3. Training YOLO Models (Opsional - model sudah tersedia)
```bash
python train_yolo_models.py
```

## 📊 Konfigurasi Training

### YOLO Training Configuration
```python
EPOCHS = 50
DATA_CONFIG = '../data.yaml'

MODELS = {
    'YOLOv12n': {'model_path': 'yolo12n.pt', 'batch_size': 100},
    'YOLOv12s': {'model_path': 'yolo12s.pt', 'batch_size': 55}, 
    'YOLOv12m': {'model_path': 'yolo12m.pt', 'batch_size': 30},
    'YOLOv12l': {'model_path': 'yolo12l.pt', 'batch_size': 20},
    'YOLOv12x': {'model_path': 'yolo12x.pt', 'batch_size': 13}
}
```

### FaceNet Training Configuration
```python
EPOCHS = 50
BATCH_SIZE = 602
LEARNING_RATE = 0.001
FEATURE_DIM = 512    # InceptionResnetV1 output
HIDDEN_DIM = 256     # Hidden layer
```

## 🎯 Model Classes

### YOLO Classes (5 classes)
- `dimas` - Wajah Dimas
- `fabian` - Wajah Fabian
- `people face` - Wajah umum
- `sendy` - Wajah Sendy
- `syahrul` - Wajah Syahrul

### FaceNet Classes (5 classes)
- `dimas` - Wajah Dimas
- `fabian` - Wajah Fabian
- `sendy` - Wajah Sendy
- `syahrul` - Wajah Syahrul
- `unknown` - Wajah tidak dikenal (dari "people face")

## 📈 Training Results

### Expected YOLO Performance
| Model | mAP@0.5 | mAP@0.5:0.95 | Model Size | Training Time |
|-------|---------|--------------|------------|---------------|
| YOLOv12n | ~0.945 | ~0.830 | 5.3MB | ~30 min |
| YOLOv12s | ~0.95 | ~0.84 | 54MB | ~45 min |
| YOLOv12m | ~0.96 | ~0.85 | 39MB | ~60 min |
| YOLOv12l | ~0.97 | ~0.86 | 51MB | ~90 min |
| YOLOv12x | ~0.98 | ~0.87 | 114MB | ~120 min |

### Expected FaceNet Performance
- **Validation Accuracy**: ~98.5%
- **Test Accuracy**: ~95.7%
- **Training Time**: ~30-60 menit (50 epochs)

## 🔧 Troubleshooting

### GPU Memory Issues
```bash
# Kurangi batch size untuk FaceNet
python train_facenet.py --batch-size 32

# Force CPU jika GPU tidak cukup
python train_facenet.py --force-cpu
```

### Dataset Issues
```bash
# Cek dataset sebelum training
python ../check_dataset.py

# Preprocessing ulang jika ada masalah
python preprocess_for_facenet.py --data ../data.yaml
```

### Model Loading Issues
```bash
# Cek model yang tersedia
ls ../models/facenet_models/latest/

# Gunakan training spesifik
ls ../models/facenet_models/training_*/
```

## 📁 Output Files

### Training Outputs
```
../models/facenet_models/
├── training_20250101_120000/    # Timestamped training
├── training_20250101_140000/    # Another training
└── latest/                      # Symlink to latest

../runs/detect/                  # YOLO training results
├── YOLOv12n_epoch50_batch100/
├── YOLOv12s_epoch50_batch55/
└── ...

../model_comparison_results.csv  # YOLO comparison
../model_comparison_charts.png   # YOLO charts
```

## 🚀 Quick Start

### Training FaceNet (Recommended)
```bash
cd training

# 1. Preprocessing (jika belum)
python preprocess_for_facenet.py

# 2. Training FaceNet
python train_facenet.py --epochs 50

# 3. Cek hasil
ls ../models/facenet_models/latest/
```

### Training YOLO (Optional)
```bash
cd training

# Training semua model YOLO
python train_yolo_models.py

# Cek hasil
ls ../runs/detect/
```

## 📞 Support

Jika mengalami masalah:
1. Pastikan dataset sudah dipreprocess dengan `preprocess_for_facenet.py`
2. Cek ketersediaan GPU dengan `nvidia-smi`
3. Gunakan `--force-cpu` jika GPU tidak tersedia
4. Periksa log error untuk debugging

---

**🎯 Happy Training!** 🏋️ 