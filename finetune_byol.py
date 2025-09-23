import os
import torch
import argparse
from byol.model import create_byol_model
from byol.finetuner import LinearFinetuner
from byol.data import create_clfdata_loaders


def main():
    parser = argparse.ArgumentParser(description="Linear Evaluation of BYOL Backbone")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--ssl_checkpoint",
        type=str,
        default="outputs/byol/best_model.pth",
        help="Path to BYOL checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/imgs/train",
        help="Path to labeled training images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/byol",
        help="Directory to save finetune results",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of finetuning epochs"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Fraction of data for validation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of data loading workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume finetuning from the specified clf_checkpoint (default: False)",
    )
    parser.add_argument(
        "--clf_checkpoint",
        type=str,
        default="outputs/byol/best_finetune_model.pth",
        help="Checkpoint to load when resuming finetuning (required if --resume is specified)",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader, num_classes = create_clfdata_loaders(args)

    # add num_classes to args
    args.num_classes = num_classes

    # Load BYOL backbone
    model = create_byol_model(
        backbone_name=args.backbone, pretrained=None, momentum=0.996
    )

    ssl_checkpoint = torch.load(args.ssl_checkpoint, map_location=device)
    model.load_state_dict(ssl_checkpoint["model_state_dict"])

    finetuner = LinearFinetuner(model, train_loader, val_loader, device, args)
    finetuner.train()


if __name__ == "__main__":
    main()
