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
    def __init__(self, dataset_dir, model_save_dir='facenet_models', device=None):
        """
        Initialize FaceNet trainer
        
        Args:
            dataset_dir: Path ke dataset yang sudah dipreprocess
            model_save_dir: Directory untuk menyimpan model
            device: Device untuk training (cuda/cpu)
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((160, 160)),
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
            batch_size=32, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        # Validation dataset
        if val_dir.exists():
            self.val_dataset = FaceNetDataset(val_dir, transform=self.val_transform)
            self.val_loader = DataLoader(
                self.val_dataset, 
                batch_size=32, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True
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
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        # Combine backbone and classifier
        self.model = nn.Sequential(
            self.backbone,
            self.classifier
        )
        
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        print(f"🤖 Model initialized with {self.num_classes} classes")
        print(f"📐 Feature dimension: 512 -> 256 -> {self.num_classes}")
    
    def train_epoch(self):
        """Train satu epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False, ncols=100, ascii=True)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
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
                data, target = data.to(self.device), target.to(self.device)
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
    
    def train(self, epochs=20):
        """Training loop utama"""
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
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model('best_facenet.pth', epoch, val_acc)
                print(f"✅ New best model saved! Accuracy: {val_acc:.2f}%")
            
            # Save checkpoint setiap 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_model(f'facenet_epoch_{epoch+1}.pth', epoch, val_acc)
        
        # Save final model
        self.save_model('final_facenet.pth', epochs-1, val_acc)
        
        # Plot training curves
        self.plot_training_curves(train_losses, train_accs, val_losses, val_accs)
        
        # Generate classification report
        self.generate_classification_report(val_preds, val_targets)
        
        print(f"\n🎉 Training selesai! Best validation accuracy: {best_acc:.2f}%")
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

def main():
    parser = argparse.ArgumentParser(description='Fine-tune FaceNet untuk face recognition')
    parser.add_argument('--dataset', default='facenet_dataset', help='Path ke dataset yang sudah dipreprocess')
    parser.add_argument('--output', default='facenet_models', help='Output directory untuk model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--device', default=None, help='Device untuk training (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset directory tidak ditemukan: {args.dataset}")
        print("💡 Jalankan preprocess_for_facenet.py terlebih dahulu!")
        return
    
    # Initialize trainer
    trainer = FaceNetTrainer(
        dataset_dir=args.dataset,
        model_save_dir=args.output,
        device=args.device
    )
    
    # Start training
    best_accuracy = trainer.train(epochs=args.epochs)
    
    print(f"\n🎯 Training Summary:")
    print(f"   📊 Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"   💾 Models saved in: {args.output}")
    print(f"   📈 Training curves: {args.output}/training_curves.png")

if __name__ == "__main__":
    main() 