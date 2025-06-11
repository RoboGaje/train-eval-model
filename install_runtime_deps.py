#!/usr/bin/env python3
"""
Script untuk menginstall dependencies runtime engines
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Jalankan command dengan error handling"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} berhasil")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} gagal: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Install runtime dependencies"""
    print("🚀 Installing Runtime Dependencies for YOLO Benchmark")
    print("="*60)
    
    # List dependencies
    dependencies = [
        ("pip install onnxruntime-gpu", "Installing ONNX Runtime GPU"),
        ("pip install tensorflow", "Installing TensorFlow"),
        ("pip install openvino", "Installing OpenVINO"),
        ("pip install tensorrt", "Installing TensorRT (might fail if not available)"),
    ]
    
    successful = 0
    failed = 0
    
    for command, description in dependencies:
        if run_command(command, description):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📦 Total: {len(dependencies)}")
    
    if failed > 0:
        print(f"\n⚠️  Some installations failed. You can still run benchmark")
        print(f"    with available engines.")
    
    print(f"\n🎉 Setup completed!")
    print(f"📝 Run: python benchmark_runtime.py")

if __name__ == "__main__":
    main() 