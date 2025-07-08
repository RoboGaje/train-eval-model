#!/usr/bin/env python3
"""
Script untuk preprocessing dataset YOLO menjadi dataset FaceNet
Mengkonversi bounding box annotations menjadi cropped face images
"""

import os
import cv2
import yaml
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

class YOLOToFaceNetPreprocessor:
    def __init__(self, data_config='data.yaml', output_dir='facenet_dataset'):
        """
        Initialize preprocessor
        
        Args:
            data_config: Path ke file data.yaml
            output_dir: Directory output untuk dataset FaceNet
        """
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        
        # Load dataset config
        with open(data_config, 'r') as f:
            self.data = yaml.safe_load(f)
        
        self.class_names = self.data['names']
        print(f"📊 Kelas yang ditemukan: {self.class_names}")
        
        # Mapping class names untuk FaceNet (exclude 'people face')
        self.facenet_classes = {}
        for idx, name in enumerate(self.class_names):
            if name != 'people face':  # Skip generic face class
                self.facenet_classes[idx] = name
        
        print(f"🎯 Kelas untuk FaceNet: {list(self.facenet_classes.values())}")
        
        # Create output directories
        self.create_output_structure()
    
    def create_output_structure(self):
        """Membuat struktur direktori output"""
        self.output_dir.mkdir(exist_ok=True)
        
        # Create directories for each split
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Create class directories
            for class_name in self.facenet_classes.values():
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
            
            # Create unknown directory
            unknown_dir = split_dir / 'unknown'
            unknown_dir.mkdir(exist_ok=True)
        
        print(f"📁 Struktur direktori dibuat di: {self.output_dir}")
    
    def crop_face_from_bbox(self, image, bbox, padding=0.2):
        """
        Crop wajah dari bounding box dengan padding
        
        Args:
            image: Input image (numpy array)
            bbox: Bounding box dalam format YOLO (x_center, y_center, width, height) normalized
            padding: Padding tambahan di sekitar bounding box
        
        Returns:
            Cropped face image
        """
        h, w = image.shape[:2]
        
        # Convert YOLO format to pixel coordinates
        x_center, y_center, width, height = bbox
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        
        # Add padding
        width_padded = width * (1 + padding)
        height_padded = height * (1 + padding)
        
        # Calculate crop coordinates
        x1 = int(x_center - width_padded / 2)
        y1 = int(y_center - height_padded / 2)
        x2 = int(x_center + width_padded / 2)
        y2 = int(y_center + height_padded / 2)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        # Resize to standard size for FaceNet (160x160)
        if face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (160, 160))
        
        return face_crop
    
    def process_split(self, split_name):
        """
        Process satu split (train/val/test)
        
        Args:
            split_name: Nama split ('train', 'val', 'test')
        """
        print(f"\n🔄 Processing {split_name} split...")
        
        # Get paths
        if split_name == 'val':
            images_path = Path(self.data['val'].replace('../', ''))
        elif split_name == 'test':
            images_path = Path(self.data['test'].replace('../', ''))
        else:  # train
            images_path = Path(self.data['train'].replace('../', ''))
        
        labels_path = Path(str(images_path).replace('images', 'labels'))
        
        if not images_path.exists() or not labels_path.exists():
            print(f"⚠️  Path tidak ditemukan untuk {split_name}")
            return
        
        # Get all image files
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.jpeg')) + list(images_path.glob('*.png'))
        
        stats = {
            'total_images': 0,
            'total_faces': 0,
            'faces_per_class': {name: 0 for name in self.facenet_classes.values()},
            'unknown_faces': 0
        }
        
        for img_file in tqdm(image_files, desc=f"Processing {split_name}", leave=False, ncols=100, ascii=True):
            # Corresponding label file
            label_file = labels_path / (img_file.stem + '.txt')
            
            if not label_file.exists():
                continue
            
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            stats['total_images'] += 1
            
            # Read labels
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            face_count = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:]]
                
                # Crop face
                face_crop = self.crop_face_from_bbox(image, bbox)
                
                if face_crop.size == 0:
                    continue
                
                # Determine output class
                if class_id in self.facenet_classes:
                    class_name = self.facenet_classes[class_id]
                    output_dir = self.output_dir / split_name / class_name
                    stats['faces_per_class'][class_name] += 1
                else:
                    # 'people face' or other -> unknown
                    class_name = 'unknown'
                    output_dir = self.output_dir / split_name / 'unknown'
                    stats['unknown_faces'] += 1
                
                # Save cropped face
                face_filename = f"{img_file.stem}_face_{face_count:02d}.jpg"
                face_path = output_dir / face_filename
                cv2.imwrite(str(face_path), face_crop)
                
                face_count += 1
                stats['total_faces'] += 1
        
        # Print statistics
        print(f"\n📊 Statistik {split_name}:")
        print(f"   📸 Total gambar: {stats['total_images']}")
        print(f"   👤 Total wajah: {stats['total_faces']}")
        print(f"   ❓ Unknown faces: {stats['unknown_faces']}")
        for class_name, count in stats['faces_per_class'].items():
            print(f"   👤 {class_name}: {count} wajah")
    
    def process_all_splits(self):
        """Process semua splits"""
        print("🚀 Memulai preprocessing dataset untuk FaceNet...")
        
        splits = []
        if 'train' in self.data:
            splits.append('train')
        if 'val' in self.data:
            splits.append('val')
        if 'test' in self.data:
            splits.append('test')
        
        for split in splits:
            self.process_split(split)
        
        print(f"\n✅ Preprocessing selesai! Dataset FaceNet tersimpan di: {self.output_dir}")
        
        # Create summary
        self.create_dataset_summary()
    
    def create_dataset_summary(self):
        """Membuat ringkasan dataset"""
        summary_file = self.output_dir / 'dataset_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("FACENET DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Dataset Structure:\n")
            f.write(f"📁 {self.output_dir}/\n")
            
            for split in ['train', 'val', 'test']:
                split_dir = self.output_dir / split
                if split_dir.exists():
                    f.write(f"├── {split}/\n")
                    
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            count = len(list(class_dir.glob('*.jpg')))
                            f.write(f"│   ├── {class_dir.name}/ ({count} images)\n")
            
            f.write(f"\nClasses for Recognition:\n")
            for class_name in self.facenet_classes.values():
                f.write(f"- {class_name}\n")
            f.write("- unknown (untuk wajah yang tidak dikenal)\n")
        
        print(f"📄 Summary tersimpan di: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess YOLO dataset untuk FaceNet')
    parser.add_argument('--data', default='../data.yaml', help='Path ke data.yaml')
    parser.add_argument('--output', default='../facenet_dataset', help='Output directory')
    parser.add_argument('--padding', type=float, default=0.2, help='Padding untuk crop (default: 0.2)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = YOLOToFaceNetPreprocessor(
        data_config=args.data,
        output_dir=args.output
    )
    
    # Process all splits
    preprocessor.process_all_splits()

if __name__ == "__main__":
    main() 