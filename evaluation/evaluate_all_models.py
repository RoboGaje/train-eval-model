#!/usr/bin/env python3
"""
Script untuk mengevaluasi semua model YOLO (n, s, m, l, x) pada test set
dan membandingkan performanya menggunakan default runtime (PyTorch)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Model yang akan dievaluasi
MODELS_TO_EVALUATE = {
    'YOLOv12n': 'models/YOLO12n/weights/best.pt',
    'YOLOv12s': 'models/YOLO12s/weights/best.pt', 
    'YOLOv12m': 'models/YOLOv12m/weights/best.pt',
    'YOLOv12l': 'models/YOLOv12l/weights/best.pt',
    'YOLOv12x': 'models/YOLOv12x/weights/best.pt'
}

# Konfigurasi evaluasi
DATA_YAML = 'data.yaml'
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

def get_model_size(model_path):
    """Dapatkan ukuran file model dalam MB"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0

def evaluate_model(model_name, model_path):
    """Evaluasi satu model pada test set"""
    print(f"\n🔄 Evaluating {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file tidak ditemukan: {model_path}")
        return None
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Ukuran model
        model_size_mb = get_model_size(model_path)
        
        # Evaluasi pada validation set (karena test set biasanya tidak ada label)
        print(f"📊 Running validation...")
        start_time = time.time()
        
        # Jalankan validasi
        results = model.val(
            data=DATA_YAML,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            save=False,
            plots=False
        )
        
        eval_time = time.time() - start_time
        
        # Ekstrak metrics
        metrics = {
            'model': model_name,
            'model_size_mb': model_size_mb,
            'eval_time_sec': eval_time,
            'map50': results.box.map50,  # mAP@0.5
            'map50_95': results.box.map,  # mAP@0.5:0.95
            'precision': results.box.mp,  # Mean precision
            'recall': results.box.mr,     # Mean recall
            'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0
        }
        
        print(f"✅ {model_name} - mAP@0.5: {metrics['map50']:.3f}, mAP@0.5:0.95: {metrics['map50_95']:.3f}")
        return metrics
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
        return None

def create_comparison_report(results):
    """Buat laporan perbandingan semua model"""
    print("\n📊 Membuat laporan perbandingan...")
    
    # Buat DataFrame
    df = pd.DataFrame(results)
    df = df.round(4)
    
    # Tampilkan tabel
    print("\n" + "="*100)
    print("📊 LAPORAN EVALUASI SEMUA MODEL YOLO")
    print("="*100)
    print(f"Dataset: {DATA_YAML}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print(f"IoU Threshold: {IOU_THRESHOLD}")
    print("-"*100)
    print(df.to_string(index=False))
    print("-"*100)
    
    # Analisis performa
    print("\n🏆 RANKING BERDASARKAN METRIK:")
    
    # Ranking berdasarkan mAP@0.5
    df_sorted_map50 = df.sort_values('map50', ascending=False)
    print("\n📈 mAP@0.5 (Higher is Better):")
    for i, row in df_sorted_map50.iterrows():
        print(f"   {row.name + 1}. {row['model']}: {row['map50']:.3f}")
    
    # Ranking berdasarkan mAP@0.5:0.95
    df_sorted_map = df.sort_values('map50_95', ascending=False)
    print("\n📈 mAP@0.5:0.95 (Higher is Better):")
    for i, row in df_sorted_map.iterrows():
        print(f"   {row.name + 1}. {row['model']}: {row['map50_95']:.3f}")
    
    # Ranking berdasarkan F1-Score
    df_sorted_f1 = df.sort_values('f1_score', ascending=False)
    print("\n📈 F1-Score (Higher is Better):")
    for i, row in df_sorted_f1.iterrows():
        print(f"   {row.name + 1}. {row['model']}: {row['f1_score']:.3f}")
    
    # Efisiensi (mAP per MB)
    df['efficiency_map50'] = df['map50'] / df['model_size_mb']
    df_sorted_eff = df.sort_values('efficiency_map50', ascending=False)
    print("\n⚡ Efisiensi (mAP@0.5 per MB):")
    for i, row in df_sorted_eff.iterrows():
        print(f"   {row.name + 1}. {row['model']}: {row['efficiency_map50']:.4f}")
    
    # Simpan hasil
    df.to_csv('model_evaluation_results.csv', index=False)
    print(f"\n💾 Hasil disimpan ke: model_evaluation_results.csv")
    
    return df

def create_visualizations(df):
    """Buat visualisasi perbandingan"""
    print("\n📈 Membuat visualisasi...")
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Perbandingan Performa Model YOLO', fontsize=16, fontweight='bold')
    
    # 1. mAP@0.5 Comparison
    axes[0,0].bar(df['model'], df['map50'], color='skyblue', alpha=0.8)
    axes[0,0].set_title('mAP@0.5 Comparison')
    axes[0,0].set_ylabel('mAP@0.5')
    axes[0,0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['map50']):
        axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. mAP@0.5:0.95 Comparison
    axes[0,1].bar(df['model'], df['map50_95'], color='lightcoral', alpha=0.8)
    axes[0,1].set_title('mAP@0.5:0.95 Comparison')
    axes[0,1].set_ylabel('mAP@0.5:0.95')
    axes[0,1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['map50_95']):
        axes[0,1].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. Model Size vs Performance
    axes[0,2].scatter(df['model_size_mb'], df['map50'], s=100, alpha=0.7, color='green')
    axes[0,2].set_title('Model Size vs mAP@0.5')
    axes[0,2].set_xlabel('Model Size (MB)')
    axes[0,2].set_ylabel('mAP@0.5')
    for i, row in df.iterrows():
        axes[0,2].annotate(row['model'], (row['model_size_mb'], row['map50']), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Precision vs Recall
    axes[1,0].scatter(df['recall'], df['precision'], s=100, alpha=0.7, color='purple')
    axes[1,0].set_title('Precision vs Recall')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    for i, row in df.iterrows():
        axes[1,0].annotate(row['model'], (row['recall'], row['precision']), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 5. F1-Score Comparison
    axes[1,1].bar(df['model'], df['f1_score'], color='orange', alpha=0.8)
    axes[1,1].set_title('F1-Score Comparison')
    axes[1,1].set_ylabel('F1-Score')
    axes[1,1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['f1_score']):
        axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 6. Efficiency (mAP per MB)
    efficiency = df['map50'] / df['model_size_mb']
    axes[1,2].bar(df['model'], efficiency, color='gold', alpha=0.8)
    axes[1,2].set_title('Efficiency (mAP@0.5 per MB)')
    axes[1,2].set_ylabel('mAP@0.5 per MB')
    axes[1,2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(efficiency):
        axes[1,2].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_visualizations.png', dpi=300, bbox_inches='tight')
    print("📊 Visualisasi disimpan ke: model_comparison_visualizations.png")
    
    return fig

def main():
    """Fungsi utama"""
    print("🚀 EVALUASI SEMUA MODEL YOLO")
    print("="*60)
    
    # Cek ketersediaan data.yaml
    if not os.path.exists(DATA_YAML):
        print(f"❌ File {DATA_YAML} tidak ditemukan!")
        return
    
    # Cek ketersediaan model
    missing_models = []
    for name, path in MODELS_TO_EVALUATE.items():
        if not os.path.exists(path):
            missing_models.append(f"{name} ({path})")
    
    if missing_models:
        print("⚠️  Model file(s) tidak ditemukan:")
        for model in missing_models:
            print(f"   - {model}")
        print("Melanjutkan dengan model yang tersedia...")
    
    # Evaluasi setiap model
    results = []
    
    for model_name, model_path in MODELS_TO_EVALUATE.items():
        if os.path.exists(model_path):
            result = evaluate_model(model_name, model_path)
            if result:
                results.append(result)
        else:
            print(f"⚠️  Skip {model_name}: file tidak ditemukan")
    
    if not results:
        print("❌ Tidak ada model yang berhasil dievaluasi!")
        return
    
    # Buat laporan perbandingan
    df_results = create_comparison_report(results)
    
    # Buat visualisasi
    create_visualizations(df_results)
    
    print(f"\n🎉 Evaluasi selesai!")
    print(f"📊 Laporan: model_evaluation_results.csv")
    print(f"📈 Visualisasi: model_comparison_visualizations.png")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   🤖 Models evaluated: {len(results)}")
    print(f"   📊 Dataset: {DATA_YAML}")
    print(f"   🎯 Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   ⚙️  Confidence threshold: {CONF_THRESHOLD}")
    print(f"   🔗 IoU threshold: {IOU_THRESHOLD}")
    
    # Best model summary
    best_map50 = df_results.loc[df_results['map50'].idxmax()]
    best_efficiency = df_results.loc[(df_results['map50'] / df_results['model_size_mb']).idxmax()]
    
    print(f"\n🏆 BEST MODELS:")
    print(f"   📈 Highest mAP@0.5: {best_map50['model']} ({best_map50['map50']:.3f})")
    print(f"   ⚡ Most Efficient: {best_efficiency['model']} ({(best_efficiency['map50'] / best_efficiency['model_size_mb']):.4f} mAP/MB)")

if __name__ == "__main__":
    main() 