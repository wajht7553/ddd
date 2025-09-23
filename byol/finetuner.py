import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt


class LinearFinetuner:
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args

        # Freeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.backbone = self.model.backbone.to(device)
        self.backbone.eval()

        # Linear classifier
        # Infer feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            feat = self.backbone(dummy)
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            feat_dim = (
                feat.shape[1] if len(feat.shape) == 2 else feat.view(1, -1).shape[1]
            )
        self.classifier = nn.Linear(feat_dim, self.args.num_classes).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.classifier.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.train_losses = []
        self.val_losses = []

    def extract_features(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            return feats

    def train_epoch(self, epoch):
        self.classifier.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            feats = self.extract_features(imgs)
            logits = self.classifier(feats)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
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
        self.classifier.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Validation"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                feats = self.extract_features(imgs)
                logits = self.classifier(feats)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self):
        print(f"Starting linear evaluation for {self.args.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.args.lr}")
        print(f"Batch size: {self.args.batch_size}")

        best_val_loss = float("inf")
        start_epoch = 0

        if getattr(self.args, "resume", False):
            if not self.args.clf_checkpoint:
                raise ValueError("Checkpoint path must be specified when resuming.")
            if not os.path.isfile(self.args.clf_checkpoint):
                raise FileNotFoundError(
                    f"Checkpoint file not found: {self.args.clf_checkpoint}"
                )
            checkpoint = torch.load(self.args.clf_checkpoint, map_location=self.device)
            self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
            self.train_losses = checkpoint.get("train_losses", [])
            self.val_losses = checkpoint.get("val_losses", [])
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(
                f"Resumed from checkpoint: {self.args.clf_checkpoint}, starting at epoch {start_epoch}"
            )

        for epoch in range(start_epoch, self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            print(
                f"Epoch {epoch+1}/{self.args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint(epoch)
        print("Training completed!")
        self.plot_training_history()

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "classifier_state_dict": self.classifier.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        if is_best:
            torch.save(
                checkpoint, os.path.join(self.args.output_dir, "best_finetune_model.pth")
            )
        else:
            torch.save(
                checkpoint,
                os.path.join(
                    self.args.output_dir, f"finetune_checkpoint_epoch_{epoch+1}.pth"
                ),
            )

    def plot_training_history(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Linear Evaluation Training History")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.args.output_dir, "finetune_training_history.png"), dpi=200
        )
        plt.close()
        print(
            f"Finetune training history plot saved to {self.args.output_dir}/finetune_training_history.png"
        )
