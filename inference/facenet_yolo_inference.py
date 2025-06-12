#!/usr/bin/env python3
"""
Script untuk inference menggunakan YOLO untuk face detection dan FaceNet untuk face recognition
YOLO: Mendeteksi lokasi wajah (bounding box)
FaceNet: Mengidentifikasi siapa pemilik wajah
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import argparse
import pickle
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import time
import torch.nn as nn
from ultralytics import YOLO

class YOLOFaceNetInference:
    def __init__(self, yolo_model_path, facenet_model_path, class_mapping_path=None, device=None, confidence_threshold=0.7, yolo_conf=0.5, use_tensorrt=False):
        """
        Initialize YOLO + FaceNet inference
        
        Args:
            yolo_model_path: Path ke YOLO model untuk face detection
            facenet_model_path: Path ke FaceNet model untuk face recognition
            class_mapping_path: Path ke class mapping file FaceNet
            device: Device untuk inference
            confidence_threshold: Threshold untuk FaceNet recognition
            yolo_conf: Confidence threshold untuk YOLO detection
            use_tensorrt: Gunakan TensorRT engine untuk YOLO (lebih cepat)
        """
        self.yolo_model_path = Path(yolo_model_path)
        self.facenet_model_path = Path(facenet_model_path)
        self.confidence_threshold = confidence_threshold
        self.yolo_conf = yolo_conf
        self.use_tensorrt = use_tensorrt
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        print(f"⚡ TensorRT: {'Enabled' if use_tensorrt else 'Disabled'}")
        
        # Load YOLO model untuk face detection
        self.load_yolo_model()
        
        # Load FaceNet model untuk face recognition
        self.load_facenet_model()
        
        # Load class mapping untuk FaceNet
        if class_mapping_path:
            self.load_class_mapping(class_mapping_path)
        else:
            # Try to load from same directory as FaceNet model
            mapping_path = self.facenet_model_path.parent / 'class_mapping.pkl'
            if mapping_path.exists():
                self.load_class_mapping(mapping_path)
            else:
                print("⚠️  Class mapping tidak ditemukan, menggunakan index sebagai label")
                self.idx_to_class = {}
        
        # Transform untuk preprocessing FaceNet
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ YOLO + FaceNet inference initialized")
    
    def load_yolo_model(self):
        """Load YOLO model untuk face detection"""
        print(f"📦 Loading YOLO model: {self.yolo_model_path}")
        
        # Check if TensorRT engine exists
        if self.use_tensorrt:
            engine_path = self.yolo_model_path.parent / f"{self.yolo_model_path.stem}.engine"
            if engine_path.exists():
                print(f"⚡ Loading TensorRT engine: {engine_path}")
                self.yolo_model = YOLO(str(engine_path))
            else:
                print(f"⚠️  TensorRT engine not found: {engine_path}")
                print(f"📦 Loading regular model: {self.yolo_model_path}")
                self.yolo_model = YOLO(str(self.yolo_model_path))
        else:
            self.yolo_model = YOLO(str(self.yolo_model_path))
        
        # Get YOLO class names
        self.yolo_classes = self.yolo_model.names
        print(f"🏷️  YOLO classes: {list(self.yolo_classes.values())}")
        
        # Create mapping untuk konsistensi dengan FaceNet
        # "people face" -> "unknown" untuk konsistensi
        self.yolo_class_mapping = {}
        for idx, class_name in self.yolo_classes.items():
            if class_name.lower() in ['people face', 'people_face', 'unknown_face']:
                self.yolo_class_mapping[idx] = 'unknown'
            else:
                self.yolo_class_mapping[idx] = class_name
        
        print(f"🔄 YOLO class mapping: {self.yolo_class_mapping}")
        print("✅ YOLO model loaded")
    
    def load_facenet_model(self):
        """Load FaceNet model untuk face recognition"""
        print(f"📦 Loading FaceNet model: {self.facenet_model_path}")
        checkpoint = torch.load(self.facenet_model_path, map_location=self.device)
        
        self.num_classes = checkpoint['num_classes']
        
        # Initialize FaceNet model dengan struktur yang sama seperti training
        # Backbone: InceptionResnetV1 untuk feature extraction
        self.backbone = InceptionResnetV1(pretrained=None, classify=False)
        
        # Classifier: Custom classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        # Load weights berdasarkan struktur checkpoint
        if 'backbone_state_dict' in checkpoint and 'classifier_state_dict' in checkpoint:
            # Load dari backbone dan classifier terpisah
            backbone_state_dict = checkpoint['backbone_state_dict']
            
            # Filter out logits keys yang tidak diinginkan dari backbone
            filtered_backbone_dict = {}
            for key, value in backbone_state_dict.items():
                if not key.startswith('logits.'):
                    filtered_backbone_dict[key] = value
            
            self.backbone.load_state_dict(filtered_backbone_dict)
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("✅ Loaded FaceNet from separate backbone and classifier state dicts")
            
        elif 'model_state_dict' in checkpoint:
            # Load dari model_state_dict yang berisi Sequential model
            combined_model = nn.Sequential(self.backbone, self.classifier)
            
            try:
                combined_model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded FaceNet from combined model state dict")
            except Exception as e:
                print(f"❌ Error loading combined model: {e}")
                # Coba load manual dengan mapping keys
                self._load_model_manual_mapping(checkpoint['model_state_dict'])
        else:
            raise ValueError("❌ FaceNet checkpoint tidak memiliki state_dict yang valid")
        
        # Combine backbone dan classifier untuk inference
        self.facenet_model = nn.Sequential(self.backbone, self.classifier)
        self.facenet_model = self.facenet_model.to(self.device)
        self.facenet_model.eval()
        
        accuracy = checkpoint.get('accuracy', 0)
        print(f"🤖 FaceNet model loaded successfully!")
        print(f"   📊 Classes: {self.num_classes}")
        print(f"   🎯 Accuracy: {accuracy:.2f}%")
    
    def _load_model_manual_mapping(self, state_dict):
        """Manual mapping untuk load model jika ada masalah struktur"""
        print("🔧 Attempting manual FaceNet loading...")
        
        # Separate backbone and classifier weights
        backbone_dict = {}
        classifier_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('0.'):  # Backbone weights (index 0 in Sequential)
                new_key = key[2:]  # Remove '0.' prefix
                backbone_dict[new_key] = value
            elif key.startswith('1.'):  # Classifier weights (index 1 in Sequential)
                new_key = key[2:]  # Remove '1.' prefix
                classifier_dict[new_key] = value
        
        # Load weights
        if backbone_dict:
            self.backbone.load_state_dict(backbone_dict)
            print("✅ Backbone weights loaded manually")
        
        if classifier_dict:
            self.classifier.load_state_dict(classifier_dict)
            print("✅ Classifier weights loaded manually")
    
    def load_class_mapping(self, mapping_path):
        """Load class mapping untuk FaceNet"""
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        
        self.class_to_idx = mapping['class_to_idx']
        self.idx_to_class = mapping['idx_to_class']
        self.class_names = mapping['class_names']
        
        print(f"🏷️  FaceNet classes loaded: {self.class_names}")
    
    def detect_and_recognize_faces(self, image):
        """
        Detect faces dengan YOLO dan recognize dengan FaceNet
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            List of (bbox, yolo_label, facenet_label, facenet_confidence) tuples
        """
        # YOLO detection
        results = self.yolo_model(image, conf=self.yolo_conf, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    yolo_conf = box.conf[0].cpu().numpy()
                    yolo_class_id = int(box.cls[0].cpu().numpy())
                    yolo_label = self.yolo_class_mapping.get(yolo_class_id, 'unknown')
                    
                    # Crop face untuk FaceNet
                    face_crop = image[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        # Convert BGR to RGB untuk FaceNet
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(face_rgb)
                        
                        # Preprocess untuk FaceNet
                        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                        
                        # FaceNet recognition
                        with torch.no_grad():
                            output = self.facenet_model(face_tensor)
                            probabilities = F.softmax(output, dim=1)
                            facenet_conf, predicted = torch.max(probabilities, 1)
                            
                            facenet_conf = facenet_conf.item()
                            predicted_idx = predicted.item()
                            
                            # Determine FaceNet name
                            if facenet_conf > self.confidence_threshold and predicted_idx in self.idx_to_class:
                                facenet_label = self.idx_to_class[predicted_idx]
                            else:
                                facenet_label = "unknown"
                            
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'yolo_label': yolo_label,
                                'yolo_conf': yolo_conf,
                                'facenet_label': facenet_label,
                                'facenet_conf': facenet_conf
                            })
        
        return detections
    
    def process_image(self, image_path, output_path=None, show_result=False):
        """
        Process single image
        
        Args:
            image_path: Path ke gambar input
            output_path: Path untuk menyimpan hasil (optional)
            show_result: Apakah menampilkan hasil
        
        Returns:
            Processed image dengan annotations
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Tidak dapat membaca gambar: {image_path}")
            return None
        
        # Detect and recognize faces
        detections = self.detect_and_recognize_faces(image)
        
        # Draw results
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            yolo_label = detection['yolo_label']
            yolo_conf = detection['yolo_conf']
            facenet_label = detection['facenet_label']
            facenet_conf = detection['facenet_conf']
            
            # Choose color based on FaceNet result
            color = (0, 255, 0) if facenet_label != "unknown" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw labels
            # YOLO label (top)
            yolo_text = f"YOLO: {yolo_label} ({yolo_conf:.2f})"
            cv2.putText(image, yolo_text, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # FaceNet label (bottom)
            facenet_text = f"FaceNet: {facenet_label} ({facenet_conf:.2f})"
            cv2.putText(image, facenet_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save result
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"💾 Hasil disimpan: {output_path}")
        
        # Show result
        if show_result:
            cv2.imshow('YOLO + FaceNet Recognition', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image, detections
    
    def process_video(self, video_path, output_path=None, show_result=False, frame_interval=1):
        """
        Process video file
        
        Args:
            video_path: Path ke video input
            output_path: Path untuk menyimpan video hasil
            show_result: Apakah menampilkan hasil real-time
            frame_interval: Interval frame untuk processing (1=setiap frame, 3=setiap 3 frame)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Tidak dapat membuka video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"🎬 Processing video: {total_frames} frames, {fps} FPS")
        print(f"📊 Frame interval: {frame_interval} (processing every {frame_interval} frame(s))")
        
        frame_count = 0
        start_time = time.time()
        last_detections = []  # Store last detections for frame interpolation
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame based on interval
            if frame_count % frame_interval == 0:
                detections = self.detect_and_recognize_faces(frame)
                last_detections = detections  # Update last detections
            else:
                # Use last detections for frames in between
                detections = last_detections
            
            # Draw results
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                yolo_label = detection['yolo_label']
                yolo_conf = detection['yolo_conf']
                facenet_label = detection['facenet_label']
                facenet_conf = detection['facenet_conf']
                
                # Choose color based on FaceNet result
                color = (0, 255, 0) if facenet_label != "unknown" else (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw labels
                # YOLO label (top)
                yolo_text = f"YOLO: {yolo_label} ({yolo_conf:.2f})"
                cv2.putText(frame, yolo_text, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # FaceNet label (bottom)
                facenet_text = f"FaceNet: {facenet_label} ({facenet_conf:.2f})"
                cv2.putText(frame, facenet_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Write frame
            if output_path:
                out.write(frame)
            
            # Show frame
            if show_result:
                cv2.imshow('YOLO + FaceNet Video Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = frame_count / total_frames * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), Time: {elapsed:.1f}s")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if show_result:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"✅ Video processing selesai: {total_time:.1f}s, {frame_count/total_time:.1f} FPS")

def main():
    parser = argparse.ArgumentParser(description='YOLO + FaceNet Inference')
    parser.add_argument('--yolo-model', required=True, help='Path ke YOLO model untuk face detection')
    parser.add_argument('--facenet-model', required=True, help='Path ke FaceNet model untuk face recognition')
    parser.add_argument('--mapping', help='Path ke class mapping file FaceNet')
    parser.add_argument('--mode', choices=['image', 'video'], default='image', help='Mode inference')
    parser.add_argument('--input', required=True, help='Path ke input file')
    parser.add_argument('--output', help='Path untuk output file')
    parser.add_argument('--facenet-conf', type=float, default=0.7, help='FaceNet confidence threshold')
    parser.add_argument('--yolo-conf', type=float, default=0.5, help='YOLO confidence threshold')
    parser.add_argument('--frame-interval', type=int, default=3, help='Frame processing interval (1=every frame, 3=every 3rd frame)')
    parser.add_argument('--show', action='store_true', help='Tampilkan hasil')
    parser.add_argument('--device', help='Device untuk inference (cuda/cpu)')
    parser.add_argument('--use-tensorrt', action='store_true', help='Gunakan TensorRT engine untuk YOLO')

    # Contoh penggunaan dari direktori inference:
    # python facenet_yolo_inference.py --yolo-model ../models/YOLO12n/weights/best.pt --facenet-model ../models/facenet_models/latest/best_facenet.pth --mapping ../models/facenet_models/latest/class_mapping.pkl --mode video --input ../WIN_20250612_17_21_33_Pro.mp4 --output ../result.mp4 --use-tensorrt
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = YOLOFaceNetInference(
        yolo_model_path=args.yolo_model,
        facenet_model_path=args.facenet_model,
        class_mapping_path=args.mapping,
        device=args.device,
        confidence_threshold=args.facenet_conf,
        yolo_conf=args.yolo_conf,
        use_tensorrt=args.use_tensorrt
    )
    
    # Run inference based on mode
    if args.mode == 'image':
        result = inference.process_image(
            image_path=args.input,
            output_path=args.output,
            show_result=args.show
        )
        
        if result:
            image, detections = result
            print(f"✅ Detected {len(detections)} faces")
            for i, detection in enumerate(detections):
                print(f"   Face {i+1}:")
                print(f"     YOLO: {detection['yolo_label']} (conf: {detection['yolo_conf']:.3f})")
                print(f"     FaceNet: {detection['facenet_label']} (conf: {detection['facenet_conf']:.3f})")
    
    elif args.mode == 'video':
        inference.process_video(
            video_path=args.input,
            output_path=args.output,
            show_result=args.show,
            frame_interval=args.frame_interval
        )

if __name__ == "__main__":
    main() 