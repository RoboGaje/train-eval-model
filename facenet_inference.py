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
        """Load trained model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.num_classes = checkpoint['num_classes']
        
        # Initialize model with new structure
        if 'backbone_state_dict' in checkpoint:
            # New structure: backbone + classifier
            self.backbone = InceptionResnetV1(pretrained=None, classify=False)
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)
            )
            
            # Load weights
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            
            # Combine for inference
            self.model = nn.Sequential(self.backbone, self.classifier)
        else:
            # Old structure: single model
            self.model = InceptionResnetV1(pretrained=None, classify=True, num_classes=self.num_classes)
            self.model.logits = nn.Linear(self.model.last_linear.in_features, self.num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"🤖 Model loaded: {self.num_classes} classes, accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
    
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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(image)
        
        results = []
        
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob > 0.9:  # Face detection confidence
                    # Crop face
                    face = self.mtcnn.extract(image, [box], save_path=None)
                    
                    if face is not None and len(face) > 0:
                        face_tensor = face[0].unsqueeze(0).to(self.device)
                        
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