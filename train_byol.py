import os
import torch
import argparse

from byol.config import load_config
from byol.trainer import BYOLTrainer
from byol.model import create_byol_model
from byol.data import create_ssldata_loaders


def main():
    parser = argparse.ArgumentParser(description="BYOL Self-Supervised Learning")

    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )

    # Allow overriding config values from command line
    parser.add_argument(
        "--train_data_dir", type=str, help="Path to training dataset directory"
    )
    parser.add_argument(
        "--val_data_dir", type=str, help="Path to validation dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "resnet34", "resnet50"],
        help="Backbone architecture",
    )
    parser.add_argument("--input_size", type=int, help="Input image size")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument(
        "--num_workers", type=int, help="Number of data loading workers"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--min_scale", type=float, help="Minimum scale for random resized crop"
    )
    parser.add_argument("--cj_prob", type=float, help="Color jitter probability")
    parser.add_argument("--cj_bright", type=float, help="Color jitter brightness")
    parser.add_argument("--cj_contrast", type=float, help="Color jitter contrast")
    parser.add_argument("--cj_sat", type=float, help="Color jitter saturation")
    parser.add_argument("--cj_hue", type=float, help="Color jitter hue")
    parser.add_argument(
        "--gaussian_blur_prob", type=float, help="Gaussian blur probability"
    )
    parser.add_argument(
        "--solarization_prob", type=float, help="Solarization probability"
    )
    parser.add_argument("--save_every", type=int, help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, help="Device to use (auto, cpu, cuda)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the specified checkpoint_dir (default: False)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint to load when resuming training (required if --resume is specified)",
    )

    cli_args = parser.parse_args()

    # Default config values in case no config file or command line arguments are provided
    default_config = {
        "train_data_dir": "data/ssl_dataset/train",
        "val_data_dir": "data/ssl_dataset/validation",
        "output_dir": "outputs/byol",
        "backbone": "resnet18",
        "input_size": 224,
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "weight_decay": 0.000001,
        "num_workers": 4,
        "seed": 42,
        "min_scale": 0.2,
        "cj_prob": 0.8,
        "cj_bright": 0.4,
        "cj_contrast": 0.4,
        "cj_sat": 0.2,
        "cj_hue": 0.1,
        "gaussian_blur_prob": 0.1,
        "solarization_prob": 0.2,
        "save_every": 10,
        "device": "auto",
        "resume": False,  # Caution: Don't change this here
        "checkpoint": None,  # Caution: When you specify a checkpoint, make sure to pass --resume flag from command line
    }

    args = load_config(default_config, cli_args)

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
    train_loader, val_loader = create_ssldata_loaders(args)

    # Create model
    model = create_byol_model(
        backbone_name=args.backbone,
        pretrained=None,  # Start from scratch for self-supervised learning
        momentum=0.996,
    )

    # Create trainer
    trainer = BYOLTrainer(model, train_loader, val_loader, device, args)
    trainer.train()


if __name__ == "__main__":
    main()
