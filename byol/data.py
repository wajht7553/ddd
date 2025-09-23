from PIL import Image
from pathlib import Path
import torch
from .transforms import get_view_transform
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from lightly.transforms.byol_transform import BYOLTransform


class SSLDataset:
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


def create_ssldata_loaders(args):
    """Create data loaders for training and validation."""

    # BYOL transform for creating two views
    # Define the base transform for each view
    view_transform = get_view_transform(args)

    transform = BYOLTransform(
        view_1_transform=view_transform,
        view_2_transform=view_transform,
    )

    # Create separate datasets for train and validation
    train_dataset = SSLDataset(root_dir=args.train_data_dir, transform=transform)
    val_dataset = SSLDataset(root_dir=args.val_data_dir, transform=transform)

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


def create_clfdata_loaders(args):
    """Create data loaders for classification fine-tuning."""

    # Data transforms (no heavy augmentation for linear eval)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(dataset.classes)