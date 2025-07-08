#!/usr/bin/env python3
"""
Script untuk body detection menggunakan YOLO12n pre-trained
Menggunakan post-processing filtering untuk mendeteksi hanya class 'person'
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from ultralytics import YOLO
import os

class BodyDetectionInference:
    def __init__(self, model_path, device=None, confidence=0.5, use_tensorrt=False):
        """
        Initialize body detection inference
        
        Args:
            model_path: Path ke YOLO model
            device: Device untuk inference
            confidence: Confidence threshold
            use_tensorrt: Flag untuk menggunakan TensorRT engine
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.use_tensorrt = use_tensorrt
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        print(f"📊 Confidence threshold: {confidence}")
        
        self.person_class_id = 0  # Index untuk 'person' di COCO
        
        # Jika flag TensorRT digunakan dan path masih .pt, cari .engine
        if self.use_tensorrt and self.model_path.suffix != '.engine':
            engine_path = self.model_path.with_suffix('.engine')
            if engine_path.exists():
                print(f"⚡ Menggunakan TensorRT engine: {engine_path.name}")
                self.model_path = engine_path
            else:
                print("⚠️  TensorRT engine tidak ditemukan, fallback ke PyTorch model .pt")
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        print("📦 Loading YOLO model...")
        self.model = YOLO(str(self.model_path))
        print("✅ Model loaded - akan filter class 'person' di post-processing")
    
    def detect_bodies(self, image):
        """
        Detect bodies dalam image
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            List of detections dengan format (x1, y1, x2, y2, confidence)
        """
        # Inference dengan filter hanya class person
        results = self.model(image, conf=self.confidence, verbose=False, classes=[0])  # Hanya class person
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Double check - hanya ambil class 'person' (index 0)
                    if class_id == self.person_class_id:
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class': 'person'
                        })
        
        return detections
    
    def process_image(self, image_path, output_path=None, show_result=False):
        """
        Process single image
        
        Args:
            image_path: Path ke gambar input
            output_path: Path untuk menyimpan hasil
            show_result: Apakah menampilkan hasil
        
        Returns:
            Processed image dengan annotations dan jumlah bodies
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Tidak dapat membaca gambar: {image_path}")
            return None, 0
        
        # Detect bodies
        start_time = time.time()
        detections = self.detect_bodies(image)
        inference_time = time.time() - start_time
        
        # Draw results
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green untuk person
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Person ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add info text
        info_text = f"Bodies: {len(detections)} | {inference_time*1000:.1f}ms"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Save result
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"💾 Hasil disimpan: {output_path}")
        
        # Show result
        if show_result:
            cv2.imshow('Body Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image, len(detections)
    
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
        total_bodies = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect bodies
            detections = self.detect_bodies(frame)
            total_bodies += len(detections)
            
            # Draw results
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']
                
                # Draw bounding box
                color = (0, 255, 0)  # Green untuk person
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"Person ({conf:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add info text
            info_text = f"Bodies: {len(detections)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Write frame
            if output_path:
                out.write(frame)
            
            # Show frame
            if show_result:
                cv2.imshow('Body Detection Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = frame_count / total_frames * 100
                avg_bodies = total_bodies / frame_count
                print(f"Progress: {progress:.1f}% | Avg bodies/frame: {avg_bodies:.1f}")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if show_result:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_bodies = total_bodies / frame_count if frame_count > 0 else 0
        print(f"✅ Video processing selesai: {total_time:.1f}s, {frame_count/total_time:.1f} FPS")
        print(f"📊 Total bodies detected: {total_bodies}, Avg per frame: {avg_bodies:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Body Detection using YOLO12n Pre-trained')
    parser.add_argument('--model', default='yolo12n.pt', help='Path ke YOLO model')
    parser.add_argument('--mode', choices=['image', 'video'], default='image', help='Mode inference')
    parser.add_argument('--input', required=True, help='Path ke input file')
    parser.add_argument('--output', help='Path untuk output file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Tampilkan hasil')
    parser.add_argument('--device', help='Device untuk inference (cuda/cpu)')
    parser.add_argument('--use-tensorrt', action='store_true', help='Gunakan TensorRT engine (.engine) jika tersedia')

    # Contoh penggunaan dari direktori inference:
    # python body_detection_inference.py --model yolo12n.pt --mode image --input ../test/images/sample.jpg --output ../body_result.jpg
    # python body_detection_inference.py --model yolo12n.pt --mode video --input ../video.mp4 --output ../body_result.mp4
    # python body_detection_inference.py --model yolo12n.engine --mode image --input sample.jpg --use-tensorrt
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BodyDetectionInference(
        model_path=args.model,
        device=args.device,
        confidence=args.confidence,
        use_tensorrt=args.use_tensorrt
    )
    
    # Run inference based on mode
    if args.mode == 'image':
        result = detector.process_image(
            image_path=args.input,
            output_path=args.output,
            show_result=args.show
        )
        
        if result:
            image, num_bodies = result
            print(f"✅ Detected {num_bodies} bodies")
    
    elif args.mode == 'video':
        detector.process_video(
            video_path=args.input,
            output_path=args.output,
            show_result=args.show
        )

if __name__ == "__main__":
    main() 