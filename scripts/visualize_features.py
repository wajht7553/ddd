import os
import random
import torch
import numpy as np
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from byol.model import create_byol_model
from PIL import Image

# Settings
VAL_DIR = "data/ssl_dataset/validation"
CHECKPOINT_PATH = "outputs/byol/best_model.pth"
N_SAMPLES = 500  # Number of images to visualize
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Collect validation image paths and labels
img_paths = []
labels = []
for fname in os.listdir(VAL_DIR):
    if fname.endswith((".jpg", ".jpeg", ".png")):
        img_paths.append(os.path.join(VAL_DIR, fname))
        # Extract class from filename, e.g., val_c0_img_656.jpg -> c0
        class_label = fname.split("_")[1]
        labels.append(class_label)

# 2. Randomly sample N_SAMPLES
if len(img_paths) > N_SAMPLES:
    idxs = random.sample(range(len(img_paths)), N_SAMPLES)
    img_paths = [img_paths[i] for i in idxs]
    labels = [labels[i] for i in idxs]

# 3. Define transform (should match your training transform except for augmentations)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 4. Load BYOL backbone
model = create_byol_model(backbone_name="resnet18", pretrained=None, momentum=0.996)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.backbone  # Use only the backbone for feature extraction
model.eval()
model.to(DEVICE)

# 5. Extract features
features = []
with torch.no_grad():
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0).to(DEVICE)
        feat = model(img)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        features.append(feat.cpu().numpy().squeeze())
features = np.stack(features)

# 6. t-SNE (3D)
tsne = TSNE(n_components=3, random_state=SEED)
features_3d = tsne.fit_transform(features)

# 7. Interactive 3D Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
classes = sorted(set(labels))
colors = plt.cm.get_cmap("tab10", len(classes))
for idx, cls in enumerate(classes):
    idxs = [i for i, l in enumerate(labels) if l == cls]
    ax.scatter(
        features_3d[idxs, 0],
        features_3d[idxs, 1],
        features_3d[idxs, 2],
        label=cls,
        alpha=0.7,
        color=colors(idx),
    )
ax.set_title("3D t-SNE of BYOL Validation Features")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.legend(title="Class")
plt.tight_layout()
plt.show()
