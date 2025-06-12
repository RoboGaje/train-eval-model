#!/usr/bin/env python3
"""
Script untuk inference menggunakan FaceNet yang sudah dilatih
Dapat digunakan untuk real-time recognition atau batch processing
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
from facenet_pytorch import InceptionResnetV1, MTCNN
import time
import torch.nn as nn

class FaceNetInference:
    def __init__(self, model_path, class_mapping_path=None, device=None, confidence_threshold=0.7):
        """
        Initialize FaceNet inference
        
        Args:
            model_path: Path ke model yang sudah dilatih
            class_mapping_path: Path ke class mapping file
            device: Device untuk inference
            confidence_threshold: Threshold untuk klasifikasi unknown
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Load model
        self.load_model()
        
        # Load class mapping
        if class_mapping_path:
            self.load_class_mapping(class_mapping_path)
        else:
            # Try to load from same directory as model
            mapping_path = self.model_path.parent / 'class_mapping.pkl'
            if mapping_path.exists():
                self.load_class_mapping(mapping_path)
            else:
                print("⚠️  Class mapping tidak ditemukan, menggunakan index sebagai label")
                self.idx_to_class = {}
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # Transform untuk preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ FaceNet inference initialized")
    
    def load_model(self):
        """Load trained FaceNet model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.num_classes = checkpoint['num_classes']
        
        # Initialize model dengan struktur yang sama seperti training
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
            print("✅ Loaded model from separate backbone and classifier state dicts")
            
        elif 'model_state_dict' in checkpoint:
            # Load dari model_state_dict yang berisi Sequential model
            combined_model = nn.Sequential(self.backbone, self.classifier)
            
            try:
                combined_model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded model from combined model state dict")
            except Exception as e:
                print(f"❌ Error loading combined model: {e}")
                # Coba load manual dengan mapping keys
                self._load_model_manual_mapping(checkpoint['model_state_dict'])
        else:
            raise ValueError("❌ Model checkpoint tidak memiliki state_dict yang valid")
        
        # Combine backbone dan classifier untuk inference
        self.model = nn.Sequential(self.backbone, self.classifier)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        accuracy = checkpoint.get('accuracy', 0)
        print(f"🤖 Model loaded successfully!")
        print(f"   📊 Classes: {self.num_classes}")
        print(f"   🎯 Accuracy: {accuracy:.2f}%")
    
    def _load_model_manual_mapping(self, state_dict):
        """Manual mapping untuk load model jika ada masalah struktur"""
        print("🔧 Attempting manual model loading...")
        
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
        """Load class mapping"""
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        
        self.class_to_idx = mapping['class_to_idx']
        self.idx_to_class = mapping['idx_to_class']
        self.class_names = mapping['class_names']
        
        print(f"🏷️  Classes loaded: {self.class_names}")
    
    def detect_and_recognize_faces(self, image):
        """
        Detect dan recognize wajah dalam gambar
        
        Args:
            image: Input image (numpy array atau PIL Image)
        
        Returns:
            List of (bbox, name, confidence) tuples
        """
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
            image_rgb = np.array(image)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(pil_image)
        
        results = []
        
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob > 0.9:  # Face detection confidence
                    # Crop face manually instead of using mtcnn.extract
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Add some padding
                    padding = 20
                    h, w = image_rgb.shape[:2]
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    # Crop face
                    face_crop = image_rgb[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        # Convert to PIL and resize
                        face_pil = Image.fromarray(face_crop)
                        face_pil = face_pil.resize((160, 160))
                        
                        # Convert to tensor
                        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                        
                        # Recognize face
                        with torch.no_grad():
                            output = self.model(face_tensor)
                            probabilities = F.softmax(output, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            
                            confidence = confidence.item()
                            predicted_idx = predicted.item()
                            
                            # Determine name
                            if confidence > self.confidence_threshold and predicted_idx in self.idx_to_class:
                                name = self.idx_to_class[predicted_idx]
                            else:
                                name = "unknown"
                            
                            results.append((box, name, confidence))
        
        return results
    
    def recognize_cropped_face(self, face_image):
        """
        Recognize wajah yang sudah di-crop
        
        Args:
            face_image: Cropped face image
        
        Returns:
            (name, confidence) tuple
        """
        if isinstance(face_image, np.ndarray):
            face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(face_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_idx = predicted.item()
            
            # Determine name
            if confidence > self.confidence_threshold and predicted_idx in self.idx_to_class:
                name = self.idx_to_class[predicted_idx]
            else:
                name = "unknown"
        
        return name, confidence
    
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
        results = self.detect_and_recognize_faces(image)
        
        # Draw results
        for box, name, confidence in results:
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save result
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"💾 Hasil disimpan: {output_path}")
        
        # Show result
        if show_result:
            cv2.imshow('FaceNet Recognition', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image, results
    
    def process_video(self, video_path, output_path=None, show_result=False):
        """
        Process video file
        
        Args:
            video_path: Path ke video input
            output_path: Path untuk menyimpan video hasil
            show_result: Apakah menampilkan hasil real-time
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
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame for speed (adjust as needed)
            if frame_count % 3 == 0:
                results = self.detect_and_recognize_faces(frame)
                
                # Draw results
                for box, name, confidence in results:
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Draw bounding box
                    color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{name} ({confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            if output_path:
                out.write(frame)
            
            # Show frame
            if show_result:
                cv2.imshow('FaceNet Video Recognition', frame)
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
    
    def real_time_recognition(self, camera_id=0):
        """
        Real-time face recognition menggunakan webcam
        
        Args:
            camera_id: ID kamera (default: 0)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Tidak dapat membuka kamera: {camera_id}")
            return
        
        print("🎥 Real-time recognition started. Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for better performance
            if frame_count % 5 == 0:
                results = self.detect_and_recognize_faces(frame)
                
                # Draw results
                for box, name, confidence in results:
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Draw bounding box
                    color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{name} ({confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate and display FPS
            if frame_count > 0 and frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('FaceNet Real-time Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                save_path = f"facenet_frame_{timestamp}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"💾 Frame saved: {save_path}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"✅ Real-time recognition stopped. Average FPS: {avg_fps:.1f}")

def main():
    parser = argparse.ArgumentParser(description='FaceNet Inference')
    parser.add_argument('--model', required=True, help='Path ke model FaceNet')
    parser.add_argument('--mapping', help='Path ke class mapping file')
    parser.add_argument('--mode', choices=['image', 'video', 'realtime'], default='image', help='Mode inference')
    parser.add_argument('--input', help='Path ke input file (untuk mode image/video)')
    parser.add_argument('--output', help='Path untuk output file')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID untuk real-time mode')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Tampilkan hasil')
    parser.add_argument('--device', help='Device untuk inference (cuda/cpu)')

    # Contoh: 
    # python facenet_inference.py --model models/facenet_models/best_facenet.pth --mapping models/facenet_models/class_mapping.pkl --mode video --input WIN_20250612_17_21_33_Pro.mp4 --output video_1_output.mp4
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = FaceNetInference(
        model_path=args.model,
        class_mapping_path=args.mapping,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    # Run inference based on mode
    if args.mode == 'image':
        if not args.input:
            print("❌ Input image path required untuk mode image")
            return
        
        result = inference.process_image(
            image_path=args.input,
            output_path=args.output,
            show_result=args.show
        )
        
        if result:
            image, detections = result
            print(f"✅ Detected {len(detections)} faces")
            for i, (box, name, conf) in enumerate(detections):
                print(f"   Face {i+1}: {name} (confidence: {conf:.3f})")
    
    elif args.mode == 'video':
        if not args.input:
            print("❌ Input video path required untuk mode video")
            return
        
        inference.process_video(
            video_path=args.input,
            output_path=args.output,
            show_result=args.show
        )
    
    elif args.mode == 'realtime':
        inference.real_time_recognition(camera_id=args.camera)

if __name__ == "__main__":
    main() 