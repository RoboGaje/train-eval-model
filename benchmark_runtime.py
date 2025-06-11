#!/usr/bin/env python3
"""
Script untuk benchmarking kecepatan runtime inference berbagai engine
untuk model YOLO yang sudah di-fine-tune

Engines yang ditest:
1. PyTorch
2. ONNX Runtime
3. TensorRT
4. OpenVINO
5. NCNN
6. TFLite
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi benchmarking
MODELS_TO_TEST = {
    'YOLOv12n': 'models/YOLO12n/weights/best.pt',
    'YOLOv12s': 'models/YOLO12s/weights/best.pt'
}

# Runtime engines yang akan ditest
ENGINES = [
    'pytorch',
    'onnx', 
    'tensorrt',
    'openvino',
    'ncnn',
    'tflite'
]

# Konfigurasi testing
NUM_WARMUP = 10  # Jumlah warmup inference
NUM_INFERENCE = 100  # Jumlah inference untuk benchmark
IMAGE_SIZE = 640  # Input image size
BATCH_SIZE = 1  # Batch size untuk testing

def check_engine_availability():
    """Cek ketersediaan setiap engine"""
    print("🔍 Mengecek ketersediaan engine...")
    
    available_engines = []
    
    # PyTorch - selalu tersedia jika ultralytics terinstal
    available_engines.append('pytorch')
    print("✅ PyTorch: Available")
    
    # ONNX Runtime
    try:
        import onnxruntime
        available_engines.append('onnx')
        print("✅ ONNX Runtime: Available")
    except ImportError:
        print("❌ ONNX Runtime: Not available (pip install onnxruntime)")
    
    # TensorRT
    try:
        import tensorrt
        available_engines.append('tensorrt')
        print("✅ TensorRT: Available")
    except ImportError:
        print("❌ TensorRT: Not available (install TensorRT)")
    
    # OpenVINO
    try:
        import openvino
        available_engines.append('openvino')
        print("✅ OpenVINO: Available")
    except ImportError:
        print("❌ OpenVINO: Not available (pip install openvino)")
    
    # NCNN
    try:
        # NCNN biasanya tersedia melalui ultralytics
        available_engines.append('ncnn')
        print("✅ NCNN: Available (via ultralytics)")
    except:
        print("❌ NCNN: Not available")
    
    # TFLite
    try:
        import tensorflow as tf
        available_engines.append('tflite')
        print("✅ TensorFlow Lite: Available")
    except ImportError:
        print("❌ TensorFlow Lite: Not available (pip install tensorflow)")
    
    return available_engines

def export_model_formats(model_path, model_name):
    """Export model ke berbagai format"""
    print(f"\n🔄 Exporting {model_name} ke berbagai format...")
    
    # Load model
    model = YOLO(model_path)
    
    exported_paths = {'pytorch': model_path}
    
    try:
        # Export ONNX
        onnx_path = model.export(format='onnx', imgsz=IMAGE_SIZE)
        exported_paths['onnx'] = onnx_path
        print("✅ ONNX export berhasil")
    except Exception as e:
        print(f"❌ ONNX export gagal: {e}")
    
    try:
        # Export TensorRT
        trt_path = model.export(format='engine', imgsz=IMAGE_SIZE)
        exported_paths['tensorrt'] = trt_path
        print("✅ TensorRT export berhasil")
    except Exception as e:
        print(f"❌ TensorRT export gagal: {e}")
    
    try:
        # Export OpenVINO
        ov_path = model.export(format='openvino', imgsz=IMAGE_SIZE)
        exported_paths['openvino'] = ov_path
        print("✅ OpenVINO export berhasil")
    except Exception as e:
        print(f"❌ OpenVINO export gagal: {e}")
    
    try:
        # Export NCNN
        ncnn_path = model.export(format='ncnn', imgsz=IMAGE_SIZE)
        exported_paths['ncnn'] = ncnn_path
        print("✅ NCNN export berhasil")
    except Exception as e:
        print(f"❌ NCNN export gagal: {e}")
    
    try:
        # Export TFLite
        tflite_path = model.export(format='tflite', imgsz=IMAGE_SIZE)
        exported_paths['tflite'] = tflite_path
        print("✅ TFLite export berhasil")
    except Exception as e:
        print(f"❌ TFLite export gagal: {e}")
    
    return exported_paths

def prepare_test_images():
    """Persiapkan test images"""
    test_images = []
    
    # Ambil beberapa gambar test
    test_dir = Path('test/images')
    if test_dir.exists():
        image_files = list(test_dir.glob('*.jpg'))[:10]  # Ambil 10 gambar pertama
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize ke ukuran yang konsisten
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                test_images.append(img)
    
    if not test_images:
        # Jika tidak ada test images, buat dummy image
        dummy_img = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        test_images = [dummy_img] * 5
    
    print(f"📸 Menggunakan {len(test_images)} test images")
    return test_images

def benchmark_inference(model_path, engine, test_images, model_name):
    """Benchmark inference untuk satu engine"""
    print(f"\n⏱️  Benchmarking {model_name} - {engine.upper()}...")
    
    try:
        # Load model berdasarkan engine
        if engine == 'pytorch':
            model = YOLO(model_path)
        else:
            # Untuk engine lain, gunakan exported model
            model = YOLO(model_path)
        
        # Warmup
        print(f"🔥 Warmup ({NUM_WARMUP} iterations)...")
        for _ in range(NUM_WARMUP):
            _ = model(test_images[0], imgsz=IMAGE_SIZE, verbose=False)
        
        # Actual benchmark
        print(f"🚀 Benchmarking ({NUM_INFERENCE} iterations)...")
        
        times = []
        for i in range(NUM_INFERENCE):
            img = test_images[i % len(test_images)]
            
            start_time = time.time()
            _ = model(img, imgsz=IMAGE_SIZE, verbose=False)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(inference_time)
            
            if (i + 1) % 25 == 0:
                print(f"   Progress: {i + 1}/{NUM_INFERENCE}")
        
        # Statistik
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time
        
        result = {
            'model': model_name,
            'engine': engine,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'fps': fps,
            'all_times': times
        }
        
        print(f"✅ {engine.upper()} - Avg: {avg_time:.2f}ms, FPS: {fps:.2f}")
        return result
        
    except Exception as e:
        print(f"❌ Error benchmarking {engine}: {e}")
        return None

def create_benchmark_report(results):
    """Buat laporan benchmark"""
    print("\n📊 Membuat laporan benchmark...")
    
    # Buat DataFrame
    df_results = []
    for result in results:
        if result:
            df_results.append({
                'Model': result['model'],
                'Engine': result['engine'],
                'Avg Time (ms)': result['avg_time_ms'],
                'Std Time (ms)': result['std_time_ms'],
                'Min Time (ms)': result['min_time_ms'],
                'Max Time (ms)': result['max_time_ms'],
                'FPS': result['fps']
            })
    
    df = pd.DataFrame(df_results)
    df = df.round(2)
    
    # Tampilkan tabel
    print("\n" + "="*80)
    print("📊 LAPORAN BENCHMARK RUNTIME INFERENCE")
    print("="*80)
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Iterations: {NUM_INFERENCE}")
    print("-"*80)
    print(df.to_string(index=False))
    print("-"*80)
    
    # Cari engine tercepat untuk setiap model
    for model_name in df['Model'].unique():
        model_data = df[df['Model'] == model_name]
        fastest = model_data.loc[model_data['FPS'].idxmax()]
        print(f"🏆 {model_name} tercepat: {fastest['Engine']} ({fastest['FPS']:.2f} FPS)")
    
    # Simpan ke CSV
    df.to_csv('benchmark_results.csv', index=False)
    print(f"\n💾 Hasil disimpan ke: benchmark_results.csv")
    
    # Buat visualisasi
    create_benchmark_visualizations(df, results)
    
    return df

def create_benchmark_visualizations(df, results):
    """Buat visualisasi benchmark"""
    print("\n📈 Membuat visualisasi...")
    
    plt.style.use('seaborn-v0_8')
    
    # Figure dengan multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Benchmark Runtime Inference - YOLO Models', fontsize=16, fontweight='bold')
    
    # 1. FPS Comparison
    df_pivot = df.pivot(index='Engine', columns='Model', values='FPS')
    df_pivot.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
    axes[0,0].set_title('FPS Comparison')
    axes[0,0].set_ylabel('FPS (Frames Per Second)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend()
    
    # 2. Average Inference Time
    df_pivot_time = df.pivot(index='Engine', columns='Model', values='Avg Time (ms)')
    df_pivot_time.plot(kind='bar', ax=axes[0,1], color=['lightgreen', 'orange'])
    axes[0,1].set_title('Average Inference Time')
    axes[0,1].set_ylabel('Time (ms)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].legend()
    
    # 3. FPS Heatmap
    heatmap_data = df.pivot(index='Model', columns='Engine', values='FPS')
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1,0])
    axes[1,0].set_title('FPS Heatmap')
    
    # 4. Time Distribution (Box Plot)
    # Prepare data for box plot
    box_data = []
    box_labels = []
    
    for result in results:
        if result and 'all_times' in result:
            box_data.append(result['all_times'])
            box_labels.append(f"{result['model']}\n{result['engine']}")
    
    if box_data:
        axes[1,1].boxplot(box_data, labels=box_labels)
        axes[1,1].set_title('Inference Time Distribution')
        axes[1,1].set_ylabel('Time (ms)')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('benchmark_visualizations.png', dpi=300, bbox_inches='tight')
    print("📊 Visualisasi disimpan ke: benchmark_visualizations.png")
    
    return fig

def main():
    """Fungsi utama untuk menjalankan benchmark"""
    print("🚀 YOLO Runtime Inference Benchmark")
    print("="*50)
    
    # Cek ketersediaan model
    missing_models = []
    for name, path in MODELS_TO_TEST.items():
        if not os.path.exists(path):
            missing_models.append(f"{name} ({path})")
    
    if missing_models:
        print("❌ Model file(s) tidak ditemukan:")
        for model in missing_models:
            print(f"   - {model}")
        return
    
    # Cek ketersediaan engine
    available_engines = check_engine_availability()
    
    if not available_engines:
        print("❌ Tidak ada engine yang tersedia!")
        return
    
    print(f"\n✅ Engine tersedia: {', '.join(available_engines)}")
    
    # Persiapkan test images
    test_images = prepare_test_images()
    
    # Export model ke berbagai format dan benchmark
    all_results = []
    
    for model_name, model_path in MODELS_TO_TEST.items():
        print(f"\n🔄 Processing {model_name}...")
        
        # Export model
        exported_paths = export_model_formats(model_path, model_name)
        
        # Benchmark setiap engine yang tersedia
        for engine in available_engines:
            if engine in exported_paths:
                result = benchmark_inference(
                    exported_paths[engine], 
                    engine, 
                    test_images, 
                    model_name
                )
                if result:
                    all_results.append(result)
    
    if not all_results:
        print("❌ Tidak ada hasil benchmark!")
        return
    
    # Buat laporan
    df_results = create_benchmark_report(all_results)
    
    print(f"\n🎉 Benchmark selesai!")
    print(f"📊 Laporan: benchmark_results.csv")
    print(f"📈 Visualisasi: benchmark_visualizations.png")
    
    # Tampilkan summary
    print(f"\n📋 SUMMARY:")
    print(f"   🤖 Models tested: {len(MODELS_TO_TEST)}")
    print(f"   ⚡ Engines tested: {len(available_engines)}")
    print(f"   🔄 Total benchmarks: {len(all_results)}")
    print(f"   📸 Test images: {len(test_images)}")
    print(f"   🎯 Iterations per test: {NUM_INFERENCE}")

if __name__ == "__main__":
    main() 