#!/usr/bin/env python3
"""
Real-time Object Tracking menggunakan YOLO dengan Webcam
Mendukung pilihan runtime: PyTorch (default) atau TensorRT (fastest)

Usage:
    python realtime_tracking.py --runtime pytorch  # Default runtime
    python realtime_tracking.py --runtime tensorrt # TensorRT (fastest)
    python realtime_tracking.py --model yolov12s   # Pilih model lain
"""

import cv2
import argparse
import time
import numpy as np
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi model yang tersedia
AVAILABLE_MODELS = {
    'yolov12n': {
        'pytorch': '../models/YOLO12n/weights/best.pt',
        'tensorrt': '../models/YOLO12n/weights/best.engine'
    },
    'yolov12s': {
        'pytorch': '../models/YOLO12s/weights/best.pt',
        'tensorrt': None  # Belum ada TensorRT untuk YOLOv12s
    },
    'yolov12m': {
        'pytorch': '../models/YOLOv12m/weights/best.pt',
        'tensorrt': None
    },
    'yolov12l': {
        'pytorch': '../models/YOLOv12l/weights/best.pt',
        'tensorrt': None
    },
    'yolov12x': {
        'pytorch': '../models/YOLOv12x/weights/best.pt',
        'tensorrt': None
    }
}

# Class names berdasarkan dataset
CLASS_NAMES = ['dimas', 'fabian', 'people face', 'sendy', 'syahrul']

# Warna untuk setiap class (BGR format)
COLORS = [
    (255, 0, 0),    # dimas - Biru
    (0, 255, 0),    # fabian - Hijau
    (0, 0, 255),    # people face - Merah
    (255, 255, 0),  # sendy - Cyan
    (255, 0, 255)   # syahrul - Magenta
]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Real-time YOLO Object Tracking')
    parser.add_argument('--model', type=str, default='yolov12n', 
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='Model to use (default: yolov12n)')
    parser.add_argument('--runtime', type=str, default='pytorch',
                       choices=['pytorch', 'tensorrt'],
                       help='Runtime engine (default: pytorch)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for inference (default: 640)')
    parser.add_argument('--show-fps', action='store_true',
                       help='Show FPS counter')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save output video to file')
    
    return parser.parse_args()

def load_model(model_name, runtime):
    """Load YOLO model dengan runtime yang dipilih"""
    print(f"🔄 Loading {model_name} with {runtime} runtime...")
    
    if runtime == 'tensorrt':
        model_path = AVAILABLE_MODELS[model_name]['tensorrt']
        if model_path is None:
            print(f"⚠️  TensorRT tidak tersedia untuk {model_name}, menggunakan PyTorch")
            model_path = AVAILABLE_MODELS[model_name]['pytorch']
            runtime = 'pytorch'
    else:
        model_path = AVAILABLE_MODELS[model_name]['pytorch']
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"🚀 Runtime: {runtime.upper()}")
        return model, runtime
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def draw_detections(frame, results, show_fps=False, fps=0):
    """Gambar bounding boxes dan labels pada frame"""
    if results and len(results) > 0:
        boxes = results[0].boxes
        
        if boxes is not None:
            for box in boxes:
                # Koordinat bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Confidence dan class
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Warna dan label
                color = COLORS[cls % len(COLORS)]
                label = f"{CLASS_NAMES[cls]}: {conf:.2f}"
                
                # Gambar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Background untuk text
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Text label
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Tampilkan FPS jika diminta
    if show_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def main():
    """Fungsi utama"""
    args = parse_arguments()
    
    print("🚀 YOLO Real-time Object Tracking")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Runtime: {args.runtime}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")
    print(f"Camera: {args.camera}")
    print(f"Image Size: {args.imgsz}")
    print("="*50)
    
    # Load model
    model, actual_runtime = load_model(args.model, args.runtime)
    if model is None:
        return
    
    # Inisialisasi webcam
    print(f"📹 Initializing camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open camera {args.camera}")
        return
    
    # Set resolusi webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Video writer jika save video diminta
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_out = 30
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps_out, frame_size)
        print(f"💾 Saving video to: {args.save_video}")
    
    print("✅ Camera initialized successfully!")
    print("\n🎮 CONTROLS:")
    print("   SPACE: Pause/Resume")
    print("   Q/ESC: Quit")
    print("   S: Save current frame")
    print("\n🔥 Starting real-time tracking...")
    
    # Variabel untuk FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    paused = False
    frame_count = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Cannot read frame from camera")
                    break
                
                # Inference
                start_time = time.time()
                results = model(frame, 
                              imgsz=args.imgsz,
                              conf=args.conf,
                              iou=args.iou,
                              verbose=False)
                inference_time = time.time() - start_time
                
                # Hitung FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Gambar detections
                frame = draw_detections(frame, results, args.show_fps, current_fps)
                
                # Info tambahan di pojok kiri bawah
                info_text = f"{args.model.upper()} | {actual_runtime.upper()} | {inference_time*1000:.1f}ms"
                cv2.putText(frame, info_text, (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Save video jika diminta
                if video_writer:
                    video_writer.write(frame)
                
                frame_count += 1
            
            # Tampilkan frame
            cv2.imshow('YOLO Real-time Tracking', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q atau ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"🎮 {status}")
            elif key == ord('s'):  # S
                filename = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Frame saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Total frames processed: {frame_count}")
        if args.save_video:
            print(f"💾 Video saved: {args.save_video}")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 