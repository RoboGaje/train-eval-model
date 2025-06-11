#!/usr/bin/env python3
"""
Script untuk mengecek statistik dataset YOLO
"""

import os
import yaml
from pathlib import Path

def check_dataset_stats(data_config='data.yaml'):
    """Mengecek statistik dataset"""
    print("📊 STATISTIK DATASET")
    print("="*50)
    
    # Baca konfigurasi data
    with open(data_config, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"📁 Konfigurasi dataset: {data_config}")
    print(f"🏷️  Jumlah kelas: {data['nc']}")
    print(f"📝 Nama kelas: {data['names']}")
    print("-"*50)
    
    # Cek setiap split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        if split in data:
            images_path = data[split].replace('../', '')
            labels_path = images_path.replace('images', 'labels')
            
            # Hitung jumlah file
            img_count = 0
            label_count = 0
            
            if os.path.exists(images_path):
                img_files = [f for f in os.listdir(images_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                img_count = len(img_files)
            
            if os.path.exists(labels_path):
                label_files = [f for f in os.listdir(labels_path) 
                             if f.lower().endswith('.txt')]
                label_count = len(label_files)
            
            print(f"📂 {split.upper()}:")
            print(f"   📸 Gambar: {img_count}")
            print(f"   🏷️  Label: {label_count}")
            print(f"   📍 Path gambar: {images_path}")
            print(f"   📍 Path label: {labels_path}")
            
            if img_count != label_count:
                print(f"   ⚠️  WARNING: Jumlah gambar dan label tidak sesuai!")
            else:
                print(f"   ✅ OK: Jumlah gambar dan label sesuai")
            print()
    
    total_images = 0
    for split in splits:
        if split in data:
            images_path = data[split].replace('../', '')
            if os.path.exists(images_path):
                img_files = [f for f in os.listdir(images_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                total_images += len(img_files)
    
    print(f"📊 RINGKASAN:")
    print(f"   🎯 Total gambar: {total_images}")
    print(f"   🏷️  Total kelas: {data['nc']}")
    print(f"   📋 Kelas: {', '.join(data['names'])}")
    
    return data

if __name__ == "__main__":
    check_dataset_stats() 