#!/usr/bin/env python3
"""
Script untuk fine-tuning dan membandingkan performa model YOLO
Model yang akan dibandingkan: YOLOv12n, YOLOv12s, YOLOv12m, YOLOv12l, YOLOv12x
"""

import os
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi training
EPOCHS = 50
DATA_CONFIG = '../data.yaml'

# Model yang akan dilatih dan dibandingkan dengan batch size yang disesuaikan
MODELS = {
    'YOLOv12n': {'model_path': 'yolo12n.pt', 'batch_size': 100},
    'YOLOv12s': {'model_path': 'yolo12s.pt', 'batch_size': 55}, 
    'YOLOv12m': {'model_path': 'yolo12m.pt', 'batch_size': 30},
    'YOLOv12l': {'model_path': 'yolo12l.pt', 'batch_size': 20},
    'YOLOv12x': {'model_path': 'yolo12x.pt', 'batch_size': 13}
}

def check_requirements():
    """Memeriksa dan menginstal requirements yang diperlukan"""
    print("🔍 Memeriksa requirements...")
    
    try:
        import ultralytics
        print(f"✅ Ultralytics sudah terinstal (versi: {ultralytics.__version__})")
    except ImportError:
        print("❌ Ultralytics belum terinstal. Silakan instal dengan: pip install ultralytics")
        return False
    
    # Cek CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA tersedia: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA tidak tersedia, akan menggunakan CPU (akan lebih lambat)")
    
    return True

def train_model(model_name, model_path, batch_size, epochs=EPOCHS):
    """Melakukan training untuk satu model"""
    print(f"\n🚀 Memulai training {model_name}...")
    print(f"   Model: {model_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Training
        results = model.train(
            data=DATA_CONFIG,
            epochs=epochs,
            batch=batch_size,
            name=f'{model_name}_epoch{epochs}_batch{batch_size}',
            patience=10,  # Early stopping patience
            save=True,
            plots=True,
            device='auto'  # Akan otomatis pilih GPU jika tersedia
        )
        
        training_time = time.time() - start_time
        
        print(f"✅ Training {model_name} selesai!")
        print(f"   Waktu training: {training_time/60:.2f} menit")
        
        # Simpan model yang sudah dilatih
        best_model_path = f"../runs/detect/{model_name}_epoch{epochs}_batch{batch_size}/weights/best.pt"
        
        return {
            'model_name': model_name,
            'training_time': training_time,
            'best_model_path': best_model_path,
            'batch_size': batch_size,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Error saat training {model_name}: {str(e)}")
        return None

def evaluate_model(model_info):
    """Evaluasi performa model pada test set"""
    if model_info is None:
        return None
        
    print(f"\n📊 Evaluasi {model_info['model_name']}...")
    
    try:
        # Load model yang sudah dilatih
        model = YOLO(model_info['best_model_path'])
        
        # Evaluasi pada test set
        metrics = model.val(data=DATA_CONFIG, split='test')
        
        return {
            'model_name': model_info['model_name'],
            'batch_size': model_info['batch_size'],
            'training_time': model_info['training_time'],
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
            'model_size_mb': os.path.getsize(model_info['best_model_path']) / (1024*1024)
        }
        
    except Exception as e:
        print(f"❌ Error saat evaluasi {model_info['model_name']}: {str(e)}")
        return None

def create_comparison_report(results):
    """Membuat laporan perbandingan performa model"""
    print("\n📈 Membuat laporan perbandingan...")
    
    # Buat DataFrame
    df = pd.DataFrame(results)
    df = df.round(4)
    
    # Tampilkan tabel perbandingan
    print("\n" + "="*80)
    print("📊 LAPORAN PERBANDINGAN PERFORMA MODEL YOLO")
    print("="*80)
    print(f"Dataset: {DATA_CONFIG}")
    print(f"Epochs: {EPOCHS}")
    print("Batch Size per model:")
    for name, info in MODELS.items():
        print(f"   - {name}: {info['batch_size']}")
    print("-"*80)
    print(df.to_string(index=False))
    print("-"*80)
    
    # Cari model terbaik
    best_map50 = df.loc[df['mAP50'].idxmax()]
    best_map50_95 = df.loc[df['mAP50-95'].idxmax()]
    fastest = df.loc[df['training_time'].idxmin()]
    smallest = df.loc[df['model_size_mb'].idxmin()]
    
    print(f"🏆 Model terbaik mAP@0.5: {best_map50['model_name']} ({best_map50['mAP50']:.4f})")
    print(f"🏆 Model terbaik mAP@0.5:0.95: {best_map50_95['model_name']} ({best_map50_95['mAP50-95']:.4f})")
    print(f"⚡ Model tercepat training: {fastest['model_name']} ({fastest['training_time']/60:.2f} menit)")
    print(f"💾 Model terkecil: {smallest['model_name']} ({smallest['model_size_mb']:.2f} MB)")
    
    # Simpan hasil ke CSV
    df.to_csv('../model_comparison_results.csv', index=False)
    print(f"\n💾 Hasil disimpan ke: ../model_comparison_results.csv")
    
    # Buat visualisasi
    create_visualizations(df)
    
    return df

def create_visualizations(df):
    """Membuat visualisasi perbandingan model"""
    print("\n📊 Membuat visualisasi...")
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Perbandingan Performa Model YOLO', fontsize=16, fontweight='bold')
    
    # 1. mAP50 Comparison
    axes[0,0].bar(df['model_name'], df['mAP50'], color='skyblue', edgecolor='navy')
    axes[0,0].set_title('mAP@0.5 Comparison')
    axes[0,0].set_ylabel('mAP@0.5')
    axes[0,0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['mAP50']):
        axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Training Time Comparison
    axes[0,1].bar(df['model_name'], df['training_time']/60, color='lightcoral', edgecolor='darkred')
    axes[0,1].set_title('Training Time Comparison')
    axes[0,1].set_ylabel('Training Time (minutes)')
    axes[0,1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['training_time']/60):
        axes[0,1].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
    
    # 3. Model Size Comparison
    axes[1,0].bar(df['model_name'], df['model_size_mb'], color='lightgreen', edgecolor='darkgreen')
    axes[1,0].set_title('Model Size Comparison')
    axes[1,0].set_ylabel('Model Size (MB)')
    axes[1,0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['model_size_mb']):
        axes[1,0].text(i, v + 2, f'{v:.1f}', ha='center', va='bottom')
    
    # 4. Precision vs Recall
    axes[1,1].scatter(df['recall'], df['precision'], s=100, alpha=0.7, c='purple')
    for i, model in enumerate(df['model_name']):
        axes[1,1].annotate(model, (df['recall'].iloc[i], df['precision'].iloc[i]), 
                          xytext=(5, 5), textcoords='offset points')
    axes[1,1].set_title('Precision vs Recall')
    axes[1,1].set_xlabel('Recall')
    axes[1,1].set_ylabel('Precision')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../model_comparison_charts.png', dpi=300, bbox_inches='tight')
    print("📊 Grafik perbandingan disimpan ke: ../model_comparison_charts.png")
    
    return fig

def main():
    """Fungsi utama untuk menjalankan fine-tuning dan perbandingan"""
    print("🤖 YOLO Model Fine-tuning & Comparison")
    print("="*50)
    
    # Cek requirements
    if not check_requirements():
        return
    
    # Cek apakah file data.yaml ada
    if not os.path.exists(DATA_CONFIG):
        print(f"❌ File {DATA_CONFIG} tidak ditemukan!")
        return
    
    # Cek apakah model files ada
    missing_models = []
    for name, info in MODELS.items():
        if not os.path.exists(info['model_path']):
            missing_models.append(f"{name} ({info['model_path']})")
    
    if missing_models:
        print("❌ Model file(s) tidak ditemukan:")
        for model in missing_models:
            print(f"   - {model}")
        return
    
    print(f"\n✅ Semua requirements dan file sudah tersedia!")
    print(f"📊 Dataset: {DATA_CONFIG}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"📦 Batch Size per model:")
    for name, info in MODELS.items():
        print(f"   - {name}: {info['batch_size']}")
    print(f"🤖 Model yang akan dilatih: {list(MODELS.keys())}\n")
    
    # Konfirmasi dari user
    response = input("Apakah Anda ingin melanjutkan fine-tuning? (y/n): ")
    if response.lower() not in ['y', 'yes', 'ya']:
        print("❌ Fine-tuning dibatalkan.")
        return
    
    # Training semua model
    training_results = []
    total_start_time = time.time()
    
    for model_name, info in MODELS.items():
        result = train_model(model_name, info['model_path'], info['batch_size'])
        if result:
            training_results.append(result)
    
    if not training_results:
        print("❌ Tidak ada model yang berhasil dilatih!")
        return
    
    # Evaluasi semua model
    print("\n🔍 Memulai evaluasi semua model...")
    evaluation_results = []
    
    for result in training_results:
        eval_result = evaluate_model(result)
        if eval_result:
            evaluation_results.append(eval_result)
    
    if not evaluation_results:
        print("❌ Tidak ada model yang berhasil dievaluasi!")
        return
    
    # Buat laporan perbandingan
    df_results = create_comparison_report(evaluation_results)
    
    total_time = time.time() - total_start_time
    print(f"\n🎉 Proses fine-tuning dan evaluasi selesai!")
    print(f"⏱️  Total waktu: {total_time/60:.2f} menit")
    print(f"📁 Hasil training tersimpan di folder: ../runs/detect/")
    print(f"📊 Laporan perbandingan: ../model_comparison_results.csv")
    print(f"📈 Grafik perbandingan: ../model_comparison_charts.png")

if __name__ == "__main__":
    main() 