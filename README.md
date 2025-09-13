# BYOL Self-Supervised Learning with LightlySSL

This repository implements Bootstrap Your Own Latent (BYOL) self-supervised learning using the LightlySSL framework for pretraining on unlabeled images.

## Overview

BYOL is a self-supervised learning algorithm that learns visual representations by predicting one augmented view of an image from another augmented view, without using negative examples. This implementation uses the LightlySSL framework to provide a clean and efficient implementation.

## Dataset

The current setup is configured for a driver behavior classification dataset with:
- **Labeled training images**: ~22,000 images across 10 classes (c0-c9)
- **SSL pretraining**: Uses only training set (~22,000 images) split into train/validation
- **No information leakage**: Test set is preserved for final evaluation only

## Files Structure

```
├── requirements.txt          # Python dependencies
├── prepare_dataset.py        # Dataset preparation script
├── byol_model.py            # BYOL model implementation
├── train_byol.py            # Training script
├── evaluate_byol.py         # Evaluation script
├── run_byol_pipeline.py     # Complete pipeline script
└── README.md               # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python run_byol_pipeline.py --epochs 100 --batch_size 64 --backbone resnet18
```

### 3. Individual Steps

#### Dataset Preparation
```bash
python prepare_dataset.py
```

#### BYOL Training
```bash
python train_byol.py \
    --train_data_dir data/ssl_dataset/train_unified \
    --val_data_dir data/ssl_dataset/val_unified \
    --output_dir outputs/byol \
    --backbone resnet18 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 3e-4
```

#### Model Evaluation
```bash
python evaluate_byol.py \
    --model_path outputs/byol/best_model.pth \
    --backbone resnet18 \
    --output_dir outputs/evaluation
```

## Key Features

### BYOL Implementation
- **Momentum-based target network**: Prevents collapse during training
- **Symmetric loss**: Predicts both views symmetrically
- **Strong augmentations**: BYOL-specific augmentation pipeline
- **Flexible backbones**: ResNet18, ResNet34, ResNet50 support

### Training Features
- **Automatic checkpointing**: Saves best model and periodic checkpoints
- **Learning rate scheduling**: Cosine annealing with warmup
- **Progress tracking**: Real-time loss monitoring
- **Visualization**: Training history plots

### Evaluation Features
- **Linear probing**: Tests frozen feature quality
- **Fine-tuning**: End-to-end evaluation
- **Comprehensive metrics**: Accuracy, confusion matrix, classification report
- **Visualization**: Confusion matrices and training curves

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 64 | Batch size for training |
| `--learning_rate` | 3e-4 | Learning rate |
| `--weight_decay` | 1e-6 | Weight decay for regularization |
| `--backbone` | resnet18 | Backbone architecture |

### BYOL-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min_scale` | 0.2 | Minimum scale for random resized crop |
| `--cj_prob` | 0.8 | Color jitter probability |
| `--cj_bright` | 0.4 | Color jitter brightness |
| `--cj_contrast` | 0.4 | Color jitter contrast |
| `--cj_sat` | 0.2 | Color jitter saturation |
| `--cj_hue` | 0.1 | Color jitter hue |
| `--gaussian_blur_prob` | 0.1 | Gaussian blur probability |
| `--solarization_prob` | 0.2 | Solarization probability |

## Understanding BYOL

### How BYOL Works

1. **Two Views**: Creates two different augmented views of the same image
2. **Online Network**: Main network that learns representations
3. **Target Network**: Momentum-updated copy of the online network
4. **Prediction**: Online network predicts target network's output
5. **Symmetric Loss**: Both views predict each other's target representation

### Key Advantages

- **No negative examples**: Unlike contrastive methods, doesn't need negative pairs
- **Stable training**: Momentum updates prevent collapse
- **Strong representations**: Learns rich visual features without labels
- **Transfer learning**: Pretrained features work well for downstream tasks

## Results Interpretation

### Training Metrics
- **Loss**: Should decrease steadily during training
- **Learning rate**: Follows cosine annealing schedule
- **Momentum updates**: Target network slowly follows online network

### Evaluation Metrics
- **Linear Probe Accuracy**: Tests quality of frozen features
- **Fine-tuning Accuracy**: Tests end-to-end performance
- **Confusion Matrix**: Shows per-class performance

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Increase number of workers or use mixed precision
3. **Poor Performance**: Check augmentation parameters or learning rate
4. **Convergence Issues**: Adjust momentum parameter or learning rate schedule

### Performance Tips

1. **Use GPU**: Training is much faster on GPU
2. **Adjust batch size**: Larger batches often lead to better performance
3. **Monitor loss**: Should decrease steadily without oscillations
4. **Check augmentations**: Strong augmentations are crucial for BYOL

## Advanced Usage

### Custom Datasets

To use with your own dataset:

1. Organize images in a single directory (no subdirectories needed)
2. Update the data path in training script
3. Adjust input size if needed
4. Modify augmentation parameters for your domain

### Hyperparameter Tuning

Key parameters to tune:
- **Learning rate**: Start with 3e-4, adjust based on loss curve
- **Momentum**: Default 0.996 works well, try 0.99-0.999
- **Augmentation strength**: Adjust based on your data characteristics
- **Batch size**: Larger is generally better if memory allows

## References

- [BYOL Paper](https://arxiv.org/abs/2006.07733): Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
- [LightlySSL Documentation](https://docs.lightly.ai/): Self-supervised learning framework
- [BYOL Implementation Guide](https://docs.lightly.ai/self-supervised-learning/examples/byol.html)

## License

This implementation is provided for educational and research purposes.
