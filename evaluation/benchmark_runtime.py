#!/usr/bin/env python3
"""
Script benchmarking sederhana untuk model YOLO yang sudah ada
Menggunakan model yang sudah di-export sebelumnya
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Model yang sudah ada berdasarkan tree output
AVAILABLE_MODELS = {
    'YOLOv12n': {
        'pytorch': 'models/YOLO12n/weights/best.pt',
        'onnx': 'models/YOLO12n/weights/best.onnx',
        'tensorrt': 'models/YOLO12n/weights/best.engine',
        'openvino': 'models/YOLO12n/weights/best_openvino_model',
        'ncnn': 'models/YOLO12n/weights/best_ncnn_model',
    },
    'YOLOv12s': {
        'pytorch': 'models/YOLO12s/weights/best.pt',
        'onnx': 'models/YOLO12s/weights/best.onnx',
        'openvino': 'models/YOLO12s/weights/best_openvino_model',
        'ncnn': 'models/YOLO12s/weights/best_ncnn_model',
        'tflite': 'models/YOLO12s/weights/best_saved_model/best_float32.tflite',
    },
    'YOLOv12m': {
        'pytorch': 'models/YOLOv12m/weights/best.pt',
        'onnx': 'models/YOLOv12m/weights/best.onnx',
        'openvino': 'models/YOLOv12m/weights/best_openvino_model',
        'ncnn': 'models/YOLOv12m/weights/best_ncnn_model',
        'tflite': 'models/YOLOv12m/weights/best_saved_model/best_float32.tflite',
    }
}

# Konfigurasi testing
NUM_WARMUP = 3
NUM_INFERENCE = 20
IMAGE_SIZE = 640

def prepare_test_image():
    """Siapkan satu test image"""
    # Coba ambil dari test directory
    test_dir = Path('test/images')
    if test_dir.exists():
        image_files = list(test_dir.glob('*.jpg'))
        if image_files:
            img = cv2.imread(str(image_files[0]))
            if img is not None:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                return img
    
    # Jika tidak ada, buat dummy image
    dummy_img = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return dummy_img

def benchmark_model_engine(model_path, engine_name, test_image, model_name):
    """Benchmark satu model dengan satu engine"""
    print(f"⏱️  Benchmarking {model_name} - {engine_name.upper()}...")
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Warmup
        for _ in range(NUM_WARMUP):
            _ = model(test_image, imgsz=IMAGE_SIZE, verbose=False)
        
        # Benchmark
        times = []
        for i in range(NUM_INFERENCE):
            start_time = time.time()
            _ = model(test_image, imgsz=IMAGE_SIZE, verbose=False)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            times.append(inference_time)
        
        # Statistik
        avg_time = np.mean(times)
        fps = 1000 / avg_time
        
        result = {
            'model': model_name,
            'engine': engine_name,
            'avg_time_ms': avg_time,
            'fps': fps
        }
        
        print(f"✅ {engine_name.upper()} - Avg: {avg_time:.2f}ms, FPS: {fps:.2f}")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Fungsi utama"""
    print("🚀 YOLO Simple Benchmark")
    print("="*50)
    
    # Siapkan test image
    test_image = prepare_test_image()
    print("📸 Test image prepared")
    
    # Hasil benchmark
    results = []
    
    # Test setiap model dan engine yang tersedia
    for model_name, engines in AVAILABLE_MODELS.items():
        print(f"\n🔄 Testing {model_name}...")
        
        for engine_name, model_path in engines.items():
            if os.path.exists(model_path):
                result = benchmark_model_engine(model_path, engine_name, test_image, model_name)
                if result:
                    results.append(result)
            else:
                print(f"⚠️  {engine_name.upper()}: File tidak ditemukan - {model_path}")
    
    if not results:
        print("❌ Tidak ada hasil benchmark!")
        return
    
    # Buat DataFrame
    df = pd.DataFrame(results)
    
    # Tampilkan hasil
    print("\n" + "="*70)
    print("📊 HASIL BENCHMARK")
    print("="*70)
    print(df.to_string(index=False))
    
    # Simpan hasil
    df.to_csv('benchmark_simple_results.csv', index=False)
    print(f"\n💾 Hasil disimpan ke: benchmark_simple_results.csv")
    
    # Cari engine tercepat untuk setiap model
    print("\n🏆 ENGINE TERCEPAT:")
    for model_name in df['model'].unique():
        model_data = df[df['model'] == model_name]
        if not model_data.empty:
            fastest = model_data.loc[model_data['fps'].idxmax()]
            print(f"   {model_name}: {fastest['engine']} ({fastest['fps']:.2f} FPS)")
    
    # Buat visualisasi sederhana
    plt.figure(figsize=(12, 6))
    
    # FPS comparison
    plt.subplot(1, 2, 1)
    df_pivot = df.pivot(index='engine', columns='model', values='fps')
    df_pivot.plot(kind='bar', ax=plt.gca())
    plt.title('FPS Comparison')
    plt.ylabel('FPS')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Time comparison
    plt.subplot(1, 2, 2)
    df_pivot_time = df.pivot(index='engine', columns='model', values='avg_time_ms')
    df_pivot_time.plot(kind='bar', ax=plt.gca())
    plt.title('Average Time Comparison')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_simple_results.png', dpi=300, bbox_inches='tight')
    print("📊 Visualisasi disimpan ke: benchmark_simple_results.png")
    
    print(f"\n🎉 Benchmark selesai! Total tests: {len(results)}")

if __name__ == "__main__":
    main() 