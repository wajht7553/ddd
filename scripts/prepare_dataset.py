import random
import shutil

from pathlib import Path


def prepare_ssl_dataset(train_val_split=0.9, seed=42):
    """
    Prepare dataset for self-supervised learning using only training images.
    Args:
        train_val_split: Fraction of data to use for training (rest for validation)
        seed: Random seed for reproducible splits
    """

    # Set random seed for reproducible splits
    random.seed(seed)

    # Define paths
    data_dir = Path("data")
    train_dir = data_dir / "imgs" / "train"
    ssl_dir = data_dir / "ssl_dataset"

    # Create SSL dataset directories
    ssl_train_dir = ssl_dir / "train"
    ssl_val_dir = ssl_dir / "validation"
    ssl_train_dir.mkdir(parents=True, exist_ok=True)
    ssl_val_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing dataset for self-supervised learning...")
    print(f"Train/Validation split: {train_val_split:.1%}/{1 - train_val_split:.1%}")

    total_count = 0
    train_count = 0
    val_count = 0

    # Process each class directory
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"Processing class {class_name}...")

            # Get all images in this class
            all_images = list(class_dir.glob("*.jpg"))
            random.shuffle(all_images)  # Shuffle for random split

            # Calculate split point
            split_point = int(len(all_images) * train_val_split)
            train_images = all_images[:split_point]
            val_images = all_images[split_point:]

            # Copy training images
            for img_file in train_images:
                new_name = f"train_{class_name}_{img_file.name}"
                shutil.copy2(img_file, ssl_train_dir / new_name)
                train_count += 1

            # Copy validation images
            for img_file in val_images:
                new_name = f"val_{class_name}_{img_file.name}"
                shutil.copy2(img_file, ssl_val_dir / new_name)
                val_count += 1

            total_count += len(all_images)
            print(
                f"  Class {class_name}: {len(train_images)} train, {len(val_images)} val"
            )

    print("\nDataset preparation complete!")
    print(f"Total images: {total_count}")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"SSL training dataset: {ssl_train_dir}")
    print(f"SSL validation dataset: {ssl_val_dir}")

    return ssl_train_dir, ssl_val_dir
