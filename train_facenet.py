#!/usr/bin/env python3
"""
Script untuk fine-tuning FaceNet untuk face recognition
Menggunakan dataset yang sudah dipreprocess dari YOLO bounding boxes
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
from PIL import Image
import seaborn as sns

# ==================== GLOBAL CONFIGURATION ====================
# Training Configuration
EPOCHS = 50
BATCH_SIZE = 602
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Model Configuration
FEATURE_DIM = 512
HIDDEN_DIM = 256
DROPOUT_RATE = 0.5
DROPOUT_RATE_2 = 0.3

# Data Configuration
IMAGE_SIZE = 160
NUM_WORKERS = 4

# GPU Configuration
FORCE_GPU = True  # Set True untuk memaksa menggunakan GPU
GPU_ID = 0        # ID GPU yang akan digunakan (jika ada multiple GPU)

# Learning Rate Scheduler
LR_STEP_SIZE = 10
LR_GAMMA = 0.1

# Save Configuration
SAVE_EVERY_N_EPOCHS = 5
# ================================================================

def check_gpu_availability():
    """Check dan setup GPU"""
    if not torch.cuda.is_available():
        if FORCE_GPU:
            raise RuntimeError("❌ GPU diperlukan tapi tidak tersedia! Set FORCE_GPU=False untuk menggunakan CPU.")
        else:
            print("⚠️  GPU tidak tersedia, menggunakan CPU")
            return torch.device('cpu')
    
    # Check GPU memory
    gpu_count = torch.cuda.device_count()
    print(f"🖥️  GPU tersedia: {gpu_count} device(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Select GPU
    if GPU_ID >= gpu_count:
        print(f"⚠️  GPU ID {GPU_ID} tidak tersedia, menggunakan GPU 0")
        device = torch.device('cuda:0')
    else:
        device = torch.device(f'cuda:{GPU_ID}')
    
    # Set default GPU
    torch.cuda.set_device(device)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    print(f"✅ Menggunakan: {device}")
    return device

class FaceNetDataset(Dataset):
    """Custom dataset untuk FaceNet training"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all image paths and labels
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Build class mapping
        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        for idx, class_dir in enumerate(sorted(class_dirs)):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            # Get all images in this class
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), idx))
        
        print(f"📊 Dataset loaded: {len(self.samples)} samples, {len(self.class_to_idx)} classes")
        print(f"🏷️  Classes: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FaceNetTrainer:
    def __init__(self, dataset_dir, model_save_dir='models/facenet_models', device=None):
        """
        Initialize FaceNet trainer
        
        Args:
            dataset_dir: Path ke dataset yang sudah dipreprocess
            model_save_dir: Directory untuk menyimpan model
            device: Device untuk training (akan di-override oleh check_gpu_availability)
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Set device dengan GPU checking
        self.device = check_gpu_availability()
        
        print(f"\n🔧 Training Configuration:")
        print(f"   📅 Epochs: {EPOCHS}")
        print(f"   📦 Batch Size: {BATCH_SIZE}")
        print(f"   📈 Learning Rate: {LEARNING_RATE}")
        print(f"   🎯 Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"   🧠 Architecture: {FEATURE_DIM} -> {HIDDEN_DIM} -> num_classes")
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        self.load_datasets()
        
        # Initialize model
        self.init_model()
    
    def load_datasets(self):
        """Load training dan validation datasets"""
        train_dir = self.dataset_dir / 'train'
        val_dir = self.dataset_dir / 'val'
        
        if not train_dir.exists():
            raise ValueError(f"Training directory tidak ditemukan: {train_dir}")
        
        # Training dataset
        self.train_dataset = FaceNetDataset(train_dir, transform=self.train_transform)
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Validation dataset
        if val_dir.exists():
            self.val_dataset = FaceNetDataset(val_dir, transform=self.val_transform)
            self.val_loader = DataLoader(
                self.val_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=NUM_WORKERS,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        else:
            print("⚠️  Validation directory tidak ditemukan, menggunakan training data untuk validasi")
            self.val_dataset = self.train_dataset
            self.val_loader = self.train_loader
        
        self.num_classes = len(self.train_dataset.class_to_idx)
        self.class_names = list(self.train_dataset.class_to_idx.keys())
        
        print(f"📊 Training samples: {len(self.train_dataset)}")
        print(f"📊 Validation samples: {len(self.val_dataset)}")
        print(f"🏷️  Number of classes: {self.num_classes}")
    
    def init_model(self):
        """Initialize FaceNet model"""
        # Load pre-trained FaceNet for feature extraction
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        
        # Create custom classifier
        # InceptionResnetV1 outputs 512-dimensional features
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(FEATURE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_2),
            nn.Linear(HIDDEN_DIM, self.num_classes)
        )
        
        # Combine backbone and classifier
        self.model = nn.Sequential(
            self.backbone,
            self.classifier
        )
        
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
        
        print(f"🤖 Model initialized with {self.num_classes} classes")
        print(f"📐 Feature dimension: {FEATURE_DIM} -> {HIDDEN_DIM} -> {self.num_classes}")
        
        # Print model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self):
        """Train satu epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False, ncols=100, ascii=True)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validasi model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation", leave=False, ncols=100, ascii=True):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc, all_preds, all_targets
    
    def train(self, epochs=None, test_images_dir='test/images', num_test_images=10):
        """Training loop utama"""
        if epochs is None:
            epochs = EPOCHS
            
        print(f"🚀 Memulai training untuk {epochs} epochs...")
        
        best_acc = 0.0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Learning rate scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # GPU memory info
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f"GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model('best_facenet.pth', epoch, val_acc)
                print(f"✅ New best model saved! Accuracy: {val_acc:.2f}%")
            
            # Save checkpoint setiap N epochs
            if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
                self.save_model(f'facenet_epoch_{epoch+1}.pth', epoch, val_acc)
                print(f"💾 Checkpoint saved: epoch {epoch+1}")
        
        # Save final model
        self.save_model('final_facenet.pth', epochs-1, val_acc)
        
        # Plot training curves
        self.plot_training_curves(train_losses, train_accs, val_losses, val_accs)
        
        # Generate classification report
        self.generate_classification_report(val_preds, val_targets)
        
        print(f"\n🎉 Training selesai! Best validation accuracy: {best_acc:.2f}%")
        
        # Run complete evaluation after training
        print(f"\n🚀 Memulai evaluasi lengkap model...")
        test_metrics, test_results = self.run_complete_evaluation(test_images_dir, num_test_images)
        
        return best_acc
    
    def save_model(self, filename, epoch, accuracy):
        """Save model dan metadata"""
        model_path = self.model_save_dir / filename
        
        # Save model state
        torch.save({
            'epoch': epoch,
            'backbone_state_dict': self.backbone.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_to_idx': self.train_dataset.class_to_idx,
            'idx_to_class': self.train_dataset.idx_to_class
        }, model_path)
        
        # Save class mapping separately
        class_mapping_path = self.model_save_dir / 'class_mapping.pkl'
        with open(class_mapping_path, 'wb') as f:
            pickle.dump({
                'class_to_idx': self.train_dataset.class_to_idx,
                'idx_to_class': self.train_dataset.idx_to_class,
                'class_names': self.class_names
            }, f)
    
    def plot_training_curves(self, train_losses, train_accs, val_losses, val_accs):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(train_accs, label='Training Accuracy', color='blue')
        ax2.plot(val_accs, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved to: {self.model_save_dir / 'training_curves.png'}")
    
    def generate_classification_report(self, predictions, targets):
        """Generate classification report"""
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Save report
        report_path = self.model_save_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("FACENET CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(targets, predictions, target_names=self.class_names))
        
        print(f"📄 Classification report saved to: {report_path}")

    def test_model_on_images(self, test_images_dir='test/images', num_test_images=10):
        """
        Test model pada gambar test dan simpan hasil visualisasi
        
        Args:
            test_images_dir: Directory gambar test
            num_test_images: Jumlah gambar untuk ditest
        """
        print(f"\n🧪 Testing model pada {num_test_images} gambar...")
        
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
            print(f"⚠️  Test directory tidak ditemukan: {test_dir}")
            return
        
        # Get test images
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.jpeg')) + list(test_dir.glob('*.png'))
        if len(image_files) == 0:
            print("⚠️  Tidak ada gambar test ditemukan")
            return
        
        # Select random images
        import random
        random.shuffle(image_files)
        selected_images = image_files[:min(num_test_images, len(image_files))]
        
        # Initialize MTCNN for face detection
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(
            image_size=160, 
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # Create test results directory
        test_results_dir = self.model_save_dir / 'test_results'
        test_results_dir.mkdir(exist_ok=True)
        
        results_summary = []
        
        for i, img_path in enumerate(selected_images):
            print(f"Testing image {i+1}/{len(selected_images)}: {img_path.name}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Convert to PIL for MTCNN
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Detect faces
            boxes, probs = mtcnn.detect(pil_image)
            
            detections = []
            
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
                            face_tensor = self.val_transform(face_pil).unsqueeze(0).to(self.device)
                            
                            # Recognize face
                            self.model.eval()
                            with torch.no_grad():
                                output = self.model(face_tensor)
                                probabilities = torch.nn.functional.softmax(output, dim=1)
                                confidence, predicted = torch.max(probabilities, 1)
                                
                                confidence = confidence.item()
                                predicted_idx = predicted.item()
                                
                                # Get class name
                                if predicted_idx < len(self.class_names):
                                    class_name = self.class_names[predicted_idx]
                                else:
                                    class_name = "unknown"
                                
                                detections.append({
                                    'box': box,
                                    'class': class_name,
                                    'confidence': confidence,
                                    'detection_prob': prob
                                })
            
            # Draw results on image
            result_image = image.copy()
            for detection in detections:
                box = detection['box']
                x1, y1, x2, y2 = box.astype(int)
                
                # Choose color based on class
                if detection['class'] == 'unknown':
                    color = (0, 0, 255)  # Red for unknown
                else:
                    color = (0, 255, 0)  # Green for known
                
                # Draw bounding box
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{detection['class']} ({detection['confidence']:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(result_image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                cv2.putText(result_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save result
            result_filename = f"test_result_{i+1:02d}_{img_path.stem}.jpg"
            result_path = test_results_dir / result_filename
            cv2.imwrite(str(result_path), result_image)
            
            # Add to summary
            results_summary.append({
                'image': img_path.name,
                'faces_detected': len(detections),
                'detections': detections
            })
        
        # Save results summary
        self.save_test_summary(results_summary, test_results_dir)
        
        print(f"✅ Test results saved in: {test_results_dir}")
        return results_summary
    
    def save_test_summary(self, results_summary, output_dir):
        """Save test results summary"""
        summary_file = output_dir / 'test_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("FACENET TEST RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            total_images = len(results_summary)
            total_faces = sum(result['faces_detected'] for result in results_summary)
            
            f.write(f"Total Images Tested: {total_images}\n")
            f.write(f"Total Faces Detected: {total_faces}\n")
            f.write(f"Average Faces per Image: {total_faces/total_images:.2f}\n\n")
            
            # Class distribution
            class_counts = {}
            confidence_scores = []
            
            for result in results_summary:
                f.write(f"Image: {result['image']}\n")
                f.write(f"  Faces detected: {result['faces_detected']}\n")
                
                for i, detection in enumerate(result['detections']):
                    class_name = detection['class']
                    confidence = detection['confidence']
                    
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidence_scores.append(confidence)
                    
                    f.write(f"    Face {i+1}: {class_name} (confidence: {confidence:.3f})\n")
                f.write("\n")
            
            # Statistics
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / total_faces) * 100 if total_faces > 0 else 0
                f.write(f"{class_name}: {count} faces ({percentage:.1f}%)\n")
            
            if confidence_scores:
                f.write(f"\nCONFIDENCE STATISTICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Confidence: {np.mean(confidence_scores):.3f}\n")
                f.write(f"Min Confidence: {np.min(confidence_scores):.3f}\n")
                f.write(f"Max Confidence: {np.max(confidence_scores):.3f}\n")
                f.write(f"Std Confidence: {np.std(confidence_scores):.3f}\n")
    
    def evaluate_on_test_set(self):
        """Evaluate model pada test set yang sudah dipreprocess"""
        print(f"\n📊 Evaluating model pada test set...")
        
        test_dir = self.dataset_dir / 'test'
        if not test_dir.exists():
            print("⚠️  Test directory tidak ditemukan")
            return None
        
        # Load test dataset
        test_dataset = FaceNetDataset(test_dir, transform=self.val_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"📊 Test samples: {len(test_dataset)}")
        
        # Evaluate
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        class_correct = {}
        class_total = {}
        
        # Initialize class counters
        for class_name in self.class_names:
            class_correct[class_name] = 0
            class_total[class_name] = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing", leave=False, ncols=100, ascii=True):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Per-class accuracy
                for i in range(target.size(0)):
                    true_class = self.class_names[target[i].item()]
                    pred_class = self.class_names[predicted[i].item()]
                    
                    class_total[true_class] += 1
                    if true_class == pred_class:
                        class_correct[true_class] += 1
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        # Calculate per-class metrics
        class_accuracies = {}
        for class_name in self.class_names:
            if class_total[class_name] > 0:
                class_accuracies[class_name] = 100. * class_correct[class_name] / class_total[class_name]
            else:
                class_accuracies[class_name] = 0.0
        
        # Save detailed test report
        self.save_test_evaluation_report(test_loss, test_acc, class_accuracies, class_total, all_preds, all_targets)
        
        print(f"📊 Test Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'class_accuracies': class_accuracies,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def save_test_evaluation_report(self, test_loss, test_acc, class_accuracies, class_total, predictions, targets):
        """Save detailed test evaluation report"""
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        
        report_file = self.model_save_dir / 'test_evaluation_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("FACENET TEST EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Overall Test Results:\n")
            f.write(f"  Test Loss: {test_loss:.4f}\n")
            f.write(f"  Test Accuracy: {test_acc:.2f}%\n\n")
            
            f.write("Per-Class Results:\n")
            f.write("-" * 30 + "\n")
            for class_name in self.class_names:
                f.write(f"{class_name:>12}: {class_accuracies[class_name]:>6.2f}% ({class_total[class_name]:>3} samples)\n")
            
            f.write(f"\nDetailed Classification Report:\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(targets, predictions, target_names=self.class_names))
        
        # Create confusion matrix visualization
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Test Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create accuracy comparison chart
        plt.figure(figsize=(12, 6))
        classes = list(class_accuracies.keys())
        accuracies = list(class_accuracies.values())
        
        bars = plt.bar(classes, accuracies, color=['green' if acc > 80 else 'orange' if acc > 60 else 'red' for acc in accuracies])
        plt.title('Per-Class Test Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'class_accuracy_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📄 Test evaluation report saved: {report_file}")
        print(f"📊 Confusion matrix saved: {self.model_save_dir / 'confusion_matrix_test.png'}")
        print(f"📈 Class accuracy chart saved: {self.model_save_dir / 'class_accuracy_test.png'}")
    
    def run_complete_evaluation(self, test_images_dir='test/images', num_test_images=10):
        """Run complete evaluation: test set + sample images"""
        print(f"\n🎯 Running Complete Model Evaluation...")
        
        # 1. Evaluate on preprocessed test set
        test_metrics = self.evaluate_on_test_set()
        
        # 2. Test on original images
        test_results = self.test_model_on_images(test_images_dir, num_test_images)
        
        # 3. Create summary report
        self.create_final_evaluation_summary(test_metrics, test_results)
        
        return test_metrics, test_results
    
    def create_final_evaluation_summary(self, test_metrics, test_results):
        """Create final evaluation summary"""
        summary_file = self.model_save_dir / 'final_evaluation_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("FACENET FINAL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("🎯 TRAINING CONFIGURATION:\n")
            f.write(f"  Epochs: {EPOCHS}\n")
            f.write(f"  Batch Size: {BATCH_SIZE}\n")
            f.write(f"  Learning Rate: {LEARNING_RATE}\n")
            f.write(f"  Architecture: {FEATURE_DIM} -> {HIDDEN_DIM} -> {len(self.class_names)}\n")
            f.write(f"  Classes: {', '.join(self.class_names)}\n\n")
            
            if test_metrics:
                f.write("📊 TEST SET EVALUATION:\n")
                f.write(f"  Test Accuracy: {test_metrics['test_accuracy']:.2f}%\n")
                f.write(f"  Test Loss: {test_metrics['test_loss']:.4f}\n\n")
                
                f.write("  Per-Class Accuracy:\n")
                for class_name, acc in test_metrics['class_accuracies'].items():
                    f.write(f"    {class_name}: {acc:.2f}%\n")
                f.write("\n")
            
            if test_results:
                f.write("🖼️  SAMPLE IMAGES TEST:\n")
                total_images = len(test_results)
                total_faces = sum(r['faces_detected'] for r in test_results)
                f.write(f"  Images Tested: {total_images}\n")
                f.write(f"  Faces Detected: {total_faces}\n")
                f.write(f"  Avg Faces/Image: {total_faces/total_images:.2f}\n\n")
            
            f.write("📁 OUTPUT FILES:\n")
            f.write(f"  Models: models/facenet_models/\n")
            f.write(f"  Test Results: models/facenet_models/test_results/\n")
            f.write(f"  Training Curves: models/facenet_models/training_curves.png\n")
            f.write(f"  Confusion Matrix: models/facenet_models/confusion_matrix_test.png\n")
            f.write(f"  Class Accuracy: models/facenet_models/class_accuracy_test.png\n")
        
        print(f"📋 Final evaluation summary saved: {summary_file}")

def main():
    # Override global variables jika ada argument
    global EPOCHS, BATCH_SIZE, LEARNING_RATE, FORCE_GPU, GPU_ID
    
    parser = argparse.ArgumentParser(description='Fine-tune FaceNet untuk face recognition')
    parser.add_argument('--dataset', default='facenet_dataset', help='Path ke dataset yang sudah dipreprocess')
    parser.add_argument('--output', default='models/facenet_models', help='Output directory untuk model')
    parser.add_argument('--epochs', type=int, default=None, help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None, help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None, help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--force-cpu', action='store_true', help='Force menggunakan CPU meskipun GPU tersedia')
    parser.add_argument('--gpu-id', type=int, default=None, help=f'GPU ID untuk digunakan (default: {GPU_ID})')
    parser.add_argument('--test-images-dir', default='test/images', help='Directory gambar test untuk evaluasi visual')
    parser.add_argument('--num-test-images', type=int, default=10, help='Jumlah gambar test untuk evaluasi visual')
    
    args = parser.parse_args()
    
    if args.epochs is not None:
        EPOCHS = args.epochs
        print(f"🔧 Override: Epochs = {EPOCHS}")
    
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
        print(f"🔧 Override: Batch Size = {BATCH_SIZE}")
    
    if args.lr is not None:
        LEARNING_RATE = args.lr
        print(f"🔧 Override: Learning Rate = {LEARNING_RATE}")
    
    if args.force_cpu:
        FORCE_GPU = False
        print(f"🔧 Override: Force CPU = True")
    
    if args.gpu_id is not None:
        GPU_ID = args.gpu_id
        print(f"🔧 Override: GPU ID = {GPU_ID}")
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset directory tidak ditemukan: {args.dataset}")
        print("💡 Jalankan preprocess_for_facenet.py terlebih dahulu!")
        return
    
    print(f"\n🎯 Final Configuration:")
    print(f"   📁 Dataset: {args.dataset}")
    print(f"   💾 Output: {args.output}")
    print(f"   📅 Epochs: {EPOCHS}")
    print(f"   📦 Batch Size: {BATCH_SIZE}")
    print(f"   📈 Learning Rate: {LEARNING_RATE}")
    print(f"   🖥️  Force GPU: {FORCE_GPU}")
    print(f"   🎮 GPU ID: {GPU_ID}")
    
    try:
        # Initialize trainer
        trainer = FaceNetTrainer(
            dataset_dir=args.dataset,
            model_save_dir=args.output
        )
        
        # Start training
        best_accuracy = trainer.train(
            epochs=args.epochs,
            test_images_dir=args.test_images_dir,
            num_test_images=args.num_test_images
        )
        
        print(f"\n🎯 Training Summary:")
        print(f"   📊 Best Validation Accuracy: {best_accuracy:.2f}%")
        print(f"   💾 Models saved in: {args.output}")
        print(f"   �� Training curves: {args.output}/training_curves.png")
        
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("💡 Coba gunakan --force-cpu jika GPU tidak tersedia")
    except KeyboardInterrupt:
        print("\n⚠️  Training dihentikan oleh user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 