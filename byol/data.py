from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from .transforms import get_view_transform
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


def create_data_loaders(args):
    """Create data loaders for training and validation."""

    # BYOL transform for creating two views
    # Define the base transform for each view
    view_transform = get_view_transform(args)

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
