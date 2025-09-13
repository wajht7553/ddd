"""
Training script for BYOL self-supervised learning on driver behavior dataset.
"""

import os
import torch
import argparse
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch import optim
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from byol_model import create_byol_model
from lightly.transforms.byol_transform import BYOLTransform


class UnifiedImageDataset:
    """Dataset for images in a single directory (no subdirectories)."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Get all image files
        self.image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            self.image_files.extend(list(self.root_dir.glob(ext)))

        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            # BYOLTransform returns two views as a list
            return self.transform(image)

        return image, image


class BYOLTrainer:
    """Trainer class for BYOL self-supervised learning."""

    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
        )

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}")

        for batch_idx, images in enumerate(pbar):
            # BYOL expects two views of the same image
            # The BYOLTransform automatically creates two views
            if isinstance(images, (tuple, list)) and len(images) == 2:
                x0, x1 = images
                x0, x1 = x0.to(self.device), x1.to(self.device)

                # Forward pass
                loss = self.model(x0, x1)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update momentum networks
                self.model.update_momentum()

                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Avg Loss": f"{total_loss / num_batches:.4f}",
                    }
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images in tqdm(self.val_loader, desc="Validation"):
                if isinstance(images, (tuple, list)) and len(images) == 2:
                    x0, x1 = images
                    x0, x1 = x0.to(self.device), x1.to(self.device)

                    loss = self.model(x0, x1)
                    total_loss += loss.item()
                    num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(self):
        """Main training loop."""
        print(f"Starting BYOL training for {self.args.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.args.learning_rate}")
        print(f"Batch size: {self.args.batch_size}")

        best_val_loss = float("inf")

        for epoch in range(self.args.epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            # Save checkpoint every few epochs
            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint(epoch)

        print("Training completed!")
        self.plot_training_history()

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        if is_best:
            torch.save(checkpoint, f"{self.args.output_dir}/best_model.pth")
            print(f"Best model saved at epoch {epoch + 1}")
        else:
            torch.save(
                checkpoint, f"{self.args.output_dir}/checkpoint_epoch_{epoch + 1}.pth"
            )

    def plot_training_history(self):
        """Plot training history."""

        matplotlib.use("Agg")  # Use non-interactive backend

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("BYOL Training History")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        # Only use log scale if all values are positive
        if all(loss > 0 for loss in self.train_losses):
            plt.yscale("log")
            plt.title("Training Loss (Log Scale)")
        else:
            plt.title("Training Loss (Linear Scale)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            f"{self.args.output_dir}/training_history.png", dpi=300, bbox_inches="tight"
        )
        plt.close()  # Close the figure to free memory
        print(
            f"Training history plot saved to {self.args.output_dir}/training_history.png"
        )


def create_data_loaders(args):
    """Create data loaders for training and validation."""

    # BYOL transform for creating two views
    # Define the base transform for each view
    view_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.input_size, scale=(args.min_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=args.cj_bright,
                contrast=args.cj_contrast,
                saturation=args.cj_sat,
                hue=args.cj_hue,
            )
            if args.cj_prob > 0
            else transforms.Lambda(lambda x: x),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23)], p=args.gaussian_blur_prob
            ),
            transforms.RandomApply(
                [transforms.RandomSolarize(128)], p=args.solarization_prob
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform = BYOLTransform(
        view_1_transform=view_transform,
        view_2_transform=view_transform,
    )

    # Create separate datasets for train and validation
    train_dataset = UnifiedImageDataset(
        root_dir=args.train_data_dir, transform=transform
    )
    val_dataset = UnifiedImageDataset(root_dir=args.val_data_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Total SSL dataset size: {len(train_dataset) + len(val_dataset)}")

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="BYOL Self-Supervised Learning")

    # Data arguments
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="data/ssl_dataset/train",
        help="Path to training dataset directory",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="data/ssl_dataset/validation",
        help="Path to validation dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/byol",
        help="Output directory for checkpoints and logs",
    )

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Backbone architecture",
    )
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # BYOL specific arguments
    parser.add_argument(
        "--min_scale",
        type=float,
        default=0.2,
        help="Minimum scale for random resized crop",
    )
    parser.add_argument(
        "--cj_prob", type=float, default=0.8, help="Color jitter probability"
    )
    parser.add_argument(
        "--cj_bright", type=float, default=0.4, help="Color jitter brightness"
    )
    parser.add_argument(
        "--cj_contrast", type=float, default=0.4, help="Color jitter contrast"
    )
    parser.add_argument(
        "--cj_sat", type=float, default=0.2, help="Color jitter saturation"
    )
    parser.add_argument("--cj_hue", type=float, default=0.1, help="Color jitter hue")
    parser.add_argument(
        "--gaussian_blur_prob",
        type=float,
        default=0.1,
        help="Gaussian blur probability",
    )
    parser.add_argument(
        "--solarization_prob", type=float, default=0.2, help="Solarization probability"
    )

    # Other arguments
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)

    # Create model
    model = create_byol_model(
        backbone_name=args.backbone,
        pretrained=False,  # Start from scratch for self-supervised learning
        momentum=0.996,
    )

    # Create trainer
    trainer = BYOLTrainer(model, train_loader, val_loader, device, args)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
