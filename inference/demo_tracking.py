#!/usr/bin/env python3
"""
Demo script untuk testing YOLO tracking (HEADLESS VERSION)
Mendukung: gambar test, video input, dan perbandingan runtime
Semua output di-export ke file, tidak ditampilkan di display
"""

import cv2
import os
import glob
import time
from ultralytics import YOLO
import argparse

def demo_with_test_images():
    """Demo dengan gambar-gambar test"""
    print("🖼️  Demo dengan test images")
    
    # Load model
    model = YOLO('../models/YOLO12n/weights/best.pt')
    print("✅ Model loaded")
    
    # Cari gambar test
    test_images = glob.glob('../test/images/*.jpg')
    if not test_images:
        print("❌ Tidak ada gambar test ditemukan di ../test/images/")
        return
    
    print(f"📊 Ditemukan {len(test_images)} gambar test")
    
    # Buat output directory
    output_dir = '../demo_output_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process setiap gambar
    for i, img_path in enumerate(test_images[:10]):  # Maksimal 10 gambar
        print(f"Processing {i+1}/{min(10, len(test_images))}: {os.path.basename(img_path)}")
        
        # Load dan process gambar
        img = cv2.imread(img_path)
        results = model(img, conf=0.5, verbose=False)
        
        # Gambar hasil
        annotated_img = results[0].plot()
        
        # Simpan hasil
        output_filename = f"result_{i+1:02d}_{os.path.basename(img_path)}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, annotated_img)
        
        print(f"   ✅ Saved: {output_path}")
    
    print(f"🎉 Demo selesai! Hasil disimpan di: {output_dir}")

def demo_with_video(input_video, output_video=None, runtime='pytorch'):
    """Demo dengan video file"""
    print(f"🎬 Demo dengan video: {input_video}")
    print(f"🔧 Runtime: {runtime}")
    
    # Load model berdasarkan runtime
    if runtime == 'tensorrt':
        model_path = '../models/YOLO12n/weights/best.engine'
        if not os.path.exists(model_path):
            print(f"❌ TensorRT engine tidak ditemukan: {model_path}")
            print("💡 Gunakan runtime 'pytorch' atau buat TensorRT engine terlebih dahulu")
            return
    else:
        model_path = '../models/YOLO12n/weights/best.pt'
    
    model = YOLO(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Buka video input
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {input_video}")
        return
    
    # Dapatkan properties video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    if not output_video:
        output_video = f"demo_output_{runtime}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    print(f"💾 Output video: {output_video}")
    
    # Variabel untuk tracking performa
    frame_count = 0
    total_inference_time = 0
    
    print("🔄 Processing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference dengan timing
            start_time = time.time()
            results = model(frame, conf=0.5, verbose=False)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Gambar hasil deteksi
            annotated_frame = results[0].plot()
            
            # Tambahkan info runtime dan timing
            info_text = f"{runtime.upper()} | Frame {frame_count+1}/{total_frames} | {inference_time*1000:.1f}ms"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Simpan ke video output
            video_writer.write(annotated_frame)
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_time = (total_inference_time / frame_count) * 1000
                print(f"   Progress: {progress:.1f}% | Avg inference: {avg_time:.1f}ms")
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        video_writer.release()
        
        # Summary
        if frame_count > 0:
            avg_inference_time = (total_inference_time / frame_count) * 1000
            avg_fps = frame_count / total_inference_time
            
            print(f"\n📊 SUMMARY:")
            print(f"   Frames processed: {frame_count}/{total_frames}")
            print(f"   Average inference time: {avg_inference_time:.1f}ms")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Runtime: {runtime.upper()}")
            print(f"   Output saved: {output_video}")
        
        print("✅ Video processing completed!")

def demo_tensorrt_vs_pytorch():
    """Demo perbandingan PyTorch vs TensorRT"""
    print("⚡🔥 Demo perbandingan PyTorch vs TensorRT")
    
    # Check apakah TensorRT engine ada
    pytorch_path = '../models/YOLO12n/weights/best.pt'
    tensorrt_path = '../models/YOLO12n/weights/best.engine'
    
    if not os.path.exists(pytorch_path):
        print(f"❌ PyTorch model tidak ditemukan: {pytorch_path}")
        return
    
    if not os.path.exists(tensorrt_path):
        print(f"❌ TensorRT engine tidak ditemukan: {tensorrt_path}")
        print("💡 Buat TensorRT engine terlebih dahulu dengan:")
        print(f"   yolo export model={pytorch_path} format=engine")
        return
    
    # Load kedua model
    model_pytorch = YOLO(pytorch_path)
    model_tensorrt = YOLO(tensorrt_path)
    
    print("✅ Kedua model berhasil dimuat:")
    print(f"   - PyTorch: {pytorch_path}")
    print(f"   - TensorRT: {tensorrt_path}")
    
    # Ambil satu gambar test
    test_images = glob.glob('../test/images/*.jpg')
    if not test_images:
        print("❌ Tidak ada gambar test")
        return
    
    img_path = test_images[0]
    img = cv2.imread(img_path)
    
    print(f"🖼️  Testing dengan: {os.path.basename(img_path)}")
    
    # Test PyTorch
    start = time.time()
    results_pytorch = model_pytorch(img, conf=0.5, verbose=False)
    pytorch_time = (time.time() - start) * 1000
    
    # Test TensorRT
    start = time.time()
    results_tensorrt = model_tensorrt(img, conf=0.5, verbose=False)
    tensorrt_time = (time.time() - start) * 1000
    
    print(f"\n⏱️  HASIL PERBANDINGAN:")
    print(f"   PyTorch:  {pytorch_time:.1f}ms")
    print(f"   TensorRT: {tensorrt_time:.1f}ms")
    print(f"   Speedup:  {pytorch_time/tensorrt_time:.1f}x")
    
    # Gambar hasil
    img_pytorch = results_pytorch[0].plot()
    img_tensorrt = results_tensorrt[0].plot()
    
    # Tambahkan text timing
    cv2.putText(img_pytorch, f"PyTorch: {pytorch_time:.1f}ms", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_tensorrt, f"TensorRT: {tensorrt_time:.1f}ms", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Gabungkan gambar
    combined = cv2.hconcat([img_pytorch, img_tensorrt])
    
    # Tambahkan header text
    header_height = 60
    header = cv2.copyMakeBorder(combined, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(50, 50, 50))
    cv2.putText(header, f"PyTorch vs TensorRT Comparison - Speedup: {pytorch_time/tensorrt_time:.1f}x", 
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Save hasil
    output_filename = 'comparison_pytorch_vs_tensorrt.jpg'
    cv2.imwrite(output_filename, header)
    
    # Save individual results
    cv2.imwrite('result_pytorch.jpg', img_pytorch)
    cv2.imwrite('result_tensorrt.jpg', img_tensorrt)
    
    print(f"✅ Hasil disimpan:")
    print(f"   - {output_filename}")
    print(f"   - result_pytorch.jpg")
    print(f"   - result_tensorrt.jpg")

def demo_video_comparison(input_video):
    """Demo perbandingan PyTorch vs TensorRT pada video"""
    print(f"🎬⚡ Demo perbandingan video: {input_video}")
    
    # Load kedua model
    model_pytorch = YOLO('../models/YOLO12n/weights/best.pt')
    model_tensorrt = YOLO('../models/YOLO12n/weights/best.engine')
    
    print("✅ Models loaded untuk perbandingan")
    
    # Buka video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {input_video}")
        return
    
    # Setup video writers
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_pytorch = cv2.VideoWriter('output_pytorch.mp4', fourcc, fps, (width, height))
    writer_tensorrt = cv2.VideoWriter('output_tensorrt.mp4', fourcc, fps, (width, height))
    writer_combined = cv2.VideoWriter('output_comparison.mp4', fourcc, fps, (width*2, height))
    
    frame_count = 0
    pytorch_times = []
    tensorrt_times = []
    
    print("🔄 Processing video dengan kedua runtime...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Test PyTorch
            start = time.time()
            results_pytorch = model_pytorch(frame, conf=0.5, verbose=False)
            pytorch_time = (time.time() - start) * 1000
            pytorch_times.append(pytorch_time)
            
            # Test TensorRT
            start = time.time()
            results_tensorrt = model_tensorrt(frame, conf=0.5, verbose=False)
            tensorrt_time = (time.time() - start) * 1000
            tensorrt_times.append(tensorrt_time)
            
            # Gambar hasil
            frame_pytorch = results_pytorch[0].plot()
            frame_tensorrt = results_tensorrt[0].plot()
            
            # Tambahkan info timing
            cv2.putText(frame_pytorch, f"PyTorch: {pytorch_time:.1f}ms", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_tensorrt, f"TensorRT: {tensorrt_time:.1f}ms", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Gabungkan frame untuk comparison video
            combined_frame = cv2.hconcat([frame_pytorch, frame_tensorrt])
            
            # Simpan ke video
            writer_pytorch.write(frame_pytorch)
            writer_tensorrt.write(frame_tensorrt)
            writer_combined.write(combined_frame)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"   Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted")
    
    finally:
        cap.release()
        writer_pytorch.release()
        writer_tensorrt.release()
        writer_combined.release()
        
        # Summary
        if pytorch_times and tensorrt_times:
            avg_pytorch = sum(pytorch_times) / len(pytorch_times)
            avg_tensorrt = sum(tensorrt_times) / len(tensorrt_times)
            speedup = avg_pytorch / avg_tensorrt
            
            print(f"\n📊 HASIL PERBANDINGAN VIDEO:")
            print(f"   Frames processed: {frame_count}")
            print(f"   PyTorch avg: {avg_pytorch:.1f}ms")
            print(f"   TensorRT avg: {avg_tensorrt:.1f}ms")
            print(f"   Speedup: {speedup:.1f}x")
            print(f"   Output files:")
            print(f"     - output_pytorch.mp4")
            print(f"     - output_tensorrt.mp4") 
            print(f"     - output_comparison.mp4 (side-by-side)")

def main():
    parser = argparse.ArgumentParser(description='Demo YOLO Tracking (Headless)')
    parser.add_argument('--mode', type=str, default='images',
                       choices=['images', 'video', 'compare', 'video-compare'],
                       help='Demo mode')
    parser.add_argument('--input', type=str, default=None,
                       help='Input video file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--runtime', type=str, default='pytorch',
                       choices=['pytorch', 'tensorrt'],
                       help='Runtime for video processing')
    
    args = parser.parse_args()
    
    print("🚀 YOLO Tracking Demo (Headless Version)")
    print("="*50)
    print("📁 Semua output akan disimpan ke file, tidak ditampilkan di display")
    print("="*50)
    
    if args.mode == 'images':
        demo_with_test_images()
    elif args.mode == 'video':
        if not args.input:
            print("❌ Error: --input video file required for video mode")
            print("   Example: python demo_tracking.py --mode video --input input.mp4 --runtime tensorrt")
            return
        demo_with_video(args.input, args.output, args.runtime)
    elif args.mode == 'compare':
        demo_tensorrt_vs_pytorch()
    elif args.mode == 'video-compare':
        if not args.input:
            print("❌ Error: --input video file required for video-compare mode")
            print("   Example: python demo_tracking.py --mode video-compare --input input.mp4")
            return
        demo_video_comparison(args.input)
    
    print("\n🎉 Demo completed! Check output files.")

if __name__ == "__main__":
    main() 