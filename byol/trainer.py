import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim


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
        """Main training loop with resume support."""
        print(f"Starting BYOL training for {self.args.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.args.learning_rate}")
        print(f"Batch size: {self.args.batch_size}")

        best_val_loss = float("inf")
        start_epoch = 0

        # Resume logic
        if getattr(self.args, "resume", False):
            checkpoint_path = getattr(self.args, "checkpoint", None)
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.train_losses = checkpoint.get("train_losses", [])
                self.val_losses = checkpoint.get("val_losses", [])
                start_epoch = checkpoint.get("epoch", 0) + 1
            else:
                print("No checkpoint found to resume from. Starting from scratch.")

        for epoch in range(start_epoch, self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            print(f"Epoch {epoch + 1}/{self.args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
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
