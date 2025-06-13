#!/usr/bin/env python3
"""
Script untuk deteksi wajah + body menggunakan dual YOLO models
- Face Detection: YOLO fine-tuned model
- Body Detection: YOLO pre-trained model (COCO)
- Analisis kepadatan dan statistik
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from ultralytics import YOLO
import os
import json

class FaceBodyDetectionInference:
    def __init__(self, face_model_path, body_model_path, device=None, 
                 face_confidence=0.5, body_confidence=0.5, use_tensorrt=False):
        """
        Initialize dual detection inference
        
        Args:
            face_model_path: Path ke YOLO face detection model
            body_model_path: Path ke YOLO body detection model
            device: Device untuk inference
            face_confidence: Confidence threshold untuk face detection
            body_confidence: Confidence threshold untuk body detection
            use_tensorrt: Flag untuk menggunakan TensorRT engine
        """
        self.face_model_path = Path(face_model_path)
        self.body_model_path = Path(body_model_path)
        self.face_confidence = face_confidence
        self.body_confidence = body_confidence
        self.use_tensorrt = use_tensorrt
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        print(f"👤 Face confidence threshold: {face_confidence}")
        print(f"🚶 Body confidence threshold: {body_confidence}")
        
        # Class IDs
        self.person_class_id = 0  # Index untuk 'person' di COCO
        
        # Handle TensorRT paths
        if self.use_tensorrt:
            self.face_model_path = self._get_tensorrt_path(self.face_model_path)
            self.body_model_path = self._get_tensorrt_path(self.body_model_path)
        
        # Load models
        self.load_models()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_faces': 0,
            'total_bodies': 0,
            'avg_faces_per_frame': 0,
            'avg_bodies_per_frame': 0,
            'max_faces_in_frame': 0,
            'max_bodies_in_frame': 0,
            'density_analysis': []
        }
    
    def _get_tensorrt_path(self, model_path):
        """Get TensorRT engine path if available"""
        if model_path.suffix != '.engine':
            engine_path = model_path.with_suffix('.engine')
            if engine_path.exists():
                print(f"⚡ Using TensorRT engine: {engine_path.name}")
                return engine_path
            else:
                print(f"⚠️  TensorRT engine not found for {model_path.name}, using PyTorch")
        return model_path
    
    def load_models(self):
        """Load both YOLO models"""
        print("📦 Loading Face Detection model...")
        self.face_model = YOLO(str(self.face_model_path))
        print("✅ Face model loaded")
        
        print("📦 Loading Body Detection model...")
        self.body_model = YOLO(str(self.body_model_path))
        print("✅ Body model loaded")
    
    def detect_faces(self, image):
        """Detect faces using fine-tuned YOLO model"""
        results = self.face_model(image, conf=self.face_confidence, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': class_name,
                        'class_id': class_id
                    })
        
        return detections
    
    def detect_bodies(self, image):
        """Detect bodies using pre-trained YOLO model"""
        results = self.body_model(image, conf=self.body_confidence, verbose=False, classes=[0])
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id == self.person_class_id:
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class': 'person',
                            'class_id': class_id
                        })
        
        return detections
    
    def calculate_density(self, faces, bodies, image_shape):
        """Calculate density analysis"""
        h, w = image_shape[:2]
        total_area = h * w
        
        # Calculate face density
        face_area = 0
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            face_area += (x2 - x1) * (y2 - y1)
        
        # Calculate body density
        body_area = 0
        for body in bodies:
            x1, y1, x2, y2 = body['bbox']
            body_area += (x2 - x1) * (y2 - y1)
        
        face_density = (face_area / total_area) * 100
        body_density = (body_area / total_area) * 100
        
        # Crowd level analysis
        num_people = len(bodies)
        if num_people == 0:
            crowd_level = "Empty"
        elif num_people <= 2:
            crowd_level = "Low"
        elif num_people <= 5:
            crowd_level = "Medium"
        elif num_people <= 10:
            crowd_level = "High"
        else:
            crowd_level = "Very High"
        
        return {
            'face_density': face_density,
            'body_density': body_density,
            'crowd_level': crowd_level,
            'people_count': num_people,
            'face_count': len(faces)
        }
    
    def draw_detections(self, image, faces, bodies, density_info):
        """Draw all detections and info on image"""
        # Draw faces
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            conf = face['confidence']
            class_name = face['class']
            
            # Face bounding box (Blue)
            color = (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Face label
            label = f"Face: {class_name} ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw bodies
        for body in bodies:
            x1, y1, x2, y2 = body['bbox']
            conf = body['confidence']
            
            # Body bounding box (Green)
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Body label
            label = f"Person ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y2), (x1+label_size[0], y2+label_size[1]+10), color, -1)
            cv2.putText(image, label, (x1, y2+label_size[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw statistics panel
        self._draw_stats_panel(image, faces, bodies, density_info)
        
        return image
    
    def _draw_stats_panel(self, image, faces, bodies, density_info):
        """Draw statistics panel on image"""
        h, w = image.shape[:2]
        panel_height = 120
        panel_width = 300
        
        # Create semi-transparent panel
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Add text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        texts = [
            f"Faces: {len(faces)}",
            f"Bodies: {len(bodies)}",
            f"Crowd Level: {density_info['crowd_level']}",
            f"Face Density: {density_info['face_density']:.1f}%",
            f"Body Density: {density_info['body_density']:.1f}%"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(image, text, (20, y_offset + i*20), font, font_scale, color, thickness)
    
    def process_image(self, image_path, output_path=None, show_result=False):
        """Process single image"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            return None
        
        # Detect faces and bodies
        start_time = time.time()
        faces = self.detect_faces(image)
        bodies = self.detect_bodies(image)
        inference_time = time.time() - start_time
        
        # Calculate density
        density_info = self.calculate_density(faces, bodies, image.shape)
        
        # Update statistics
        self._update_stats(faces, bodies, density_info)
        
        # Draw results
        result_image = self.draw_detections(image.copy(), faces, bodies, density_info)
        
        # Add inference time
        cv2.putText(result_image, f"Inference: {inference_time*1000:.1f}ms", 
                   (10, result_image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Save result
        if output_path:
            cv2.imwrite(str(output_path), result_image)
            print(f"💾 Result saved: {output_path}")
        
        # Show result
        if show_result:
            cv2.imshow('Face + Body Detection', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print results
        print(f"✅ Detected {len(faces)} faces, {len(bodies)} bodies")
        print(f"📊 Crowd Level: {density_info['crowd_level']}")
        print(f"📈 Densities - Face: {density_info['face_density']:.1f}%, Body: {density_info['body_density']:.1f}%")
        
        return result_image, faces, bodies, density_info
    
    def process_video(self, video_path, output_path=None, show_result=False, save_stats=False):
        """Process video file"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
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
            
            # Detect faces and bodies
            faces = self.detect_faces(frame)
            bodies = self.detect_bodies(frame)
            
            # Calculate density
            density_info = self.calculate_density(faces, bodies, frame.shape)
            
            # Update statistics
            self._update_stats(faces, bodies, density_info)
            
            # Draw results
            result_frame = self.draw_detections(frame.copy(), faces, bodies, density_info)
            
            # Write frame
            if output_path:
                out.write(result_frame)
            
            # Show frame
            if show_result:
                cv2.imshow('Face + Body Detection Video', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = frame_count / total_frames * 100
                print(f"Progress: {progress:.1f}% | Avg FPS: {frame_count/elapsed:.1f}")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if show_result:
            cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"✅ Video processing complete: {total_time:.1f}s, {frame_count/total_time:.1f} FPS")
        self._print_final_stats()
        
        # Save statistics
        if save_stats:
            stats_path = Path(output_path).with_suffix('.json') if output_path else 'video_stats.json'
            self._save_stats(stats_path)
    
    def _update_stats(self, faces, bodies, density_info):
        """Update running statistics"""
        self.stats['total_frames'] += 1
        self.stats['total_faces'] += len(faces)
        self.stats['total_bodies'] += len(bodies)
        
        self.stats['max_faces_in_frame'] = max(self.stats['max_faces_in_frame'], len(faces))
        self.stats['max_bodies_in_frame'] = max(self.stats['max_bodies_in_frame'], len(bodies))
        
        self.stats['density_analysis'].append(density_info)
        
        # Update averages
        if self.stats['total_frames'] > 0:
            self.stats['avg_faces_per_frame'] = self.stats['total_faces'] / self.stats['total_frames']
            self.stats['avg_bodies_per_frame'] = self.stats['total_bodies'] / self.stats['total_frames']
    
    def _print_final_stats(self):
        """Print final statistics"""
        print("\n📊 FINAL STATISTICS")
        print("=" * 50)
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Total Faces: {self.stats['total_faces']}")
        print(f"Total Bodies: {self.stats['total_bodies']}")
        print(f"Avg Faces/Frame: {self.stats['avg_faces_per_frame']:.2f}")
        print(f"Avg Bodies/Frame: {self.stats['avg_bodies_per_frame']:.2f}")
        print(f"Max Faces in Frame: {self.stats['max_faces_in_frame']}")
        print(f"Max Bodies in Frame: {self.stats['max_bodies_in_frame']}")
        
        # Crowd level distribution
        crowd_levels = [d['crowd_level'] for d in self.stats['density_analysis']]
        from collections import Counter
        crowd_dist = Counter(crowd_levels)
        print(f"\nCrowd Level Distribution:")
        for level, count in crowd_dist.items():
            percentage = (count / len(crowd_levels)) * 100
            print(f"  {level}: {count} frames ({percentage:.1f}%)")
    
    def _save_stats(self, stats_path):
        """Save statistics to JSON file"""
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        print(f"📊 Statistics saved: {stats_path}")

def main():
    parser = argparse.ArgumentParser(description='Face + Body Detection with Density Analysis')
    parser.add_argument('--face-model', required=True, help='Path ke YOLO face detection model')
    parser.add_argument('--body-model', default='yolo12n.pt', help='Path ke YOLO body detection model')
    parser.add_argument('--mode', choices=['image', 'video'], default='image', help='Mode inference')
    parser.add_argument('--input', required=True, help='Path ke input file')
    parser.add_argument('--output', help='Path untuk output file')
    parser.add_argument('--face-conf', type=float, default=0.5, help='Face confidence threshold')
    parser.add_argument('--body-conf', type=float, default=0.5, help='Body confidence threshold')
    parser.add_argument('--show', action='store_true', help='Tampilkan hasil')
    parser.add_argument('--device', help='Device untuk inference (cuda/cpu)')
    parser.add_argument('--use-tensorrt', action='store_true', help='Gunakan TensorRT engine jika tersedia')
    parser.add_argument('--save-stats', action='store_true', help='Simpan statistik ke file JSON')

    args = parser.parse_args()
    
    # Initialize detector
    detector = FaceBodyDetectionInference(
        face_model_path=args.face_model,
        body_model_path=args.body_model,
        device=args.device,
        face_confidence=args.face_conf,
        body_confidence=args.body_conf,
        use_tensorrt=args.use_tensorrt
    )
    
    # Run inference
    if args.mode == 'image':
        result = detector.process_image(
            image_path=args.input,
            output_path=args.output,
            show_result=args.show
        )
    
    elif args.mode == 'video':
        detector.process_video(
            video_path=args.input,
            output_path=args.output,
            show_result=args.show,
            save_stats=args.save_stats
        )

if __name__ == "__main__":
    main() 