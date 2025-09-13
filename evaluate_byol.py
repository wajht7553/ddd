"""
Evaluation script for BYOL pretrained models.
Tests the learned representations on downstream classification tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import os
from pathlib import Path

from byol_model import create_byol_model


class LinearClassifier(nn.Module):
    """Linear classifier for evaluating learned representations."""
    
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)


class BYOLEvaluator:
    """Evaluator for BYOL pretrained models."""
    
    def __init__(self, model_path, device, args):
        self.device = device
        self.args = args
        
        # Load pretrained BYOL model
        self.model = self.load_pretrained_model(model_path)
        self.model.eval()
        
    def load_pretrained_model(self, model_path):
        """Load pretrained BYOL model."""
        print(f"Loading pretrained model from {model_path}")
        
        # Create model architecture
        model = create_byol_model(
            backbone_name=self.args.backbone,
            pretrained=False
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract only the backbone for feature extraction
        backbone = model.backbone
        return backbone
    
    def extract_features(self, data_loader):
        """Extract features from the pretrained backbone."""
        features = []
        labels = []
        
        print("Extracting features...")
        with torch.no_grad():
            for images, targets in tqdm(data_loader):
                images = images.to(self.device)
                
                # Extract features
                feat = self.model(images)
                feat = feat.view(feat.size(0), -1)  # Flatten
                
                features.append(feat.cpu().numpy())
                labels.append(targets.numpy())
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return features, labels
    
    def evaluate_linear_probe(self, train_features, train_labels, val_features, val_labels):
        """Evaluate using linear probing (frozen features + linear classifier)."""
        print("Evaluating with linear probe...")
        
        # Train linear classifier
        classifier = LogisticRegression(
            random_state=self.args.seed,
            max_iter=1000,
            C=1.0
        )
        
        classifier.fit(train_features, train_labels)
        
        # Predict
        train_pred = classifier.predict(train_features)
        val_pred = classifier.predict(val_features)
        
        # Calculate accuracies
        train_acc = accuracy_score(train_labels, train_pred)
        val_acc = accuracy_score(val_labels, val_pred)
        
        print(f"Linear Probe - Train Accuracy: {train_acc:.4f}")
        print(f"Linear Probe - Validation Accuracy: {val_acc:.4f}")
        
        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'classifier': classifier,
            'val_pred': val_pred,
            'val_labels': val_labels
        }
    
    def evaluate_fine_tuning(self, train_loader, val_loader, num_classes):
        """Evaluate using fine-tuning (end-to-end training)."""
        print("Evaluating with fine-tuning...")
        
        # Create linear classifier
        feature_dim = self.get_feature_dim()
        classifier = LinearClassifier(feature_dim, num_classes).to(self.device)
        
        # Combine backbone and classifier
        full_model = nn.Sequential(self.model, classifier)
        
        # Optimizer (only for classifier initially)
        optimizer = optim.Adam(classifier.parameters(), lr=self.args.ft_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        train_accs = []
        val_accs = []
        
        for epoch in range(self.args.ft_epochs):
            # Training
            full_model.train()
            train_correct = 0
            train_total = 0
            
            for images, targets in tqdm(train_loader, desc=f'Fine-tuning Epoch {epoch+1}'):
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = full_model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            train_acc = train_correct / train_total
            train_accs.append(train_acc)
            
            # Validation
            val_acc = self.evaluate_model(full_model, val_loader)
            val_accs.append(val_acc)
            
            print(f'Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = full_model.state_dict().copy()
        
        # Load best model
        full_model.load_state_dict(best_model_state)
        
        return {
            'best_val_acc': best_val_acc,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'model': full_model
        }
    
    def evaluate_model(self, model, data_loader):
        """Evaluate model on validation set."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in data_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total
    
    def get_feature_dim(self):
        """Get feature dimension from the backbone."""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            features = self.model(dummy_input)
        return features.view(1, -1).size(1)
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self, train_accs, val_accs, save_path):
        """Plot training curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Fine-tuning Training Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_evaluation_data_loaders(args):
    """Create data loaders for evaluation."""
    
    # Standard transforms for evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load labeled training data for evaluation
    dataset = ImageFolder(root='data/imgs/train', transform=transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Evaluation dataset size: {len(dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    return train_loader, val_loader, dataset.classes


def main():
    parser = argparse.ArgumentParser(description='Evaluate BYOL Pretrained Model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained BYOL model checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Backbone architecture')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Fine-tuning arguments
    parser.add_argument('--ft_epochs', type=int, default=50,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--ft_lr', type=float, default=1e-3,
                       help='Learning rate for fine-tuning')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create evaluation data loaders
    train_loader, val_loader, class_names = create_evaluation_data_loaders(args)
    
    # Create evaluator
    evaluator = BYOLEvaluator(args.model_path, device, args)
    
    # Extract features for linear probing
    print("Extracting features for linear probing...")
    train_features, train_labels = evaluator.extract_features(train_loader)
    val_features, val_labels = evaluator.extract_features(val_loader)
    
    # Linear probe evaluation
    linear_results = evaluator.evaluate_linear_probe(
        train_features, train_labels, val_features, val_labels
    )
    
    # Fine-tuning evaluation
    fine_tuning_results = evaluator.evaluate_fine_tuning(
        train_loader, val_loader, len(class_names)
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Linear Probe - Validation Accuracy: {linear_results['val_acc']:.4f}")
    print(f"Fine-tuning - Best Validation Accuracy: {fine_tuning_results['best_val_acc']:.4f}")
    
    # Save results
    results = {
        'linear_probe': linear_results,
        'fine_tuning': fine_tuning_results,
        'class_names': class_names
    }
    
    torch.save(results, f'{args.output_dir}/evaluation_results.pth')
    
    # Plot confusion matrix for linear probe
    evaluator.plot_confusion_matrix(
        val_labels, linear_results['val_pred'], class_names,
        f'{args.output_dir}/linear_probe_confusion_matrix.png'
    )
    
    # Plot training curves for fine-tuning
    evaluator.plot_training_curves(
        fine_tuning_results['train_accs'],
        fine_tuning_results['val_accs'],
        f'{args.output_dir}/fine_tuning_curves.png'
    )
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
