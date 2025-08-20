import os, glob, warnings
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.manifold import TSNE

VAL_DIR     = "/db/shared/phenotyping/PlantNet/val"
CKPT_PATH   = "simclr_finetuned_model.pth"
IMG_SIZE    = 224
BATCH_SIZE  = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUBSAMPLE   = None       # 12000
PERPLEXITY  = 35
TSNE_ITERS  = 1000
METRIC      = "cosine"

class LabeledDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform
        class_folders = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
        if len(class_folders) == 0:
            raise RuntimeError(f"No class folders found in {root_dir}")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_folders)}
        for cls in class_folders:
            files = glob.glob(os.path.join(root_dir, cls, "*.jpg"))
            self.samples.extend([(f, self.class_to_idx[cls]) for f in files])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_ds = LabeledDataset(VAL_DIR, val_tf)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

class FineTuneModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        out = self.fc(h)
        return out

@torch.no_grad()
def extract_embeddings_and_preds(model, loader, device, max_items=None):
    model.eval()
    feats, labels, preds = [], [], []
    for imgs, y in loader:
        imgs = imgs.to(device)
        y = y.to(device)
        h = model.encoder(imgs).flatten(1)
        logits = model.fc(h)
        p = logits.argmax(1)
        feats.append(h.cpu())
        labels.append(y.cpu())
        preds.append(p.cpu())
        if max_items is not None and sum(len(a) for a in labels) >= max_items:
            break
    feats  = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    preds  = torch.cat(preds, 0)
    if max_items is not None and len(labels) > max_items:
        feats  = feats[:max_items]
        labels = labels[:max_items]
        preds  = preds[:max_items]
    return feats.numpy(), labels.numpy(), preds.numpy()

def plot_by_species(X2, y, num_classes, out_png):
    base_cmap = cm.get_cmap("nipy_spectral", num_classes)
    palette = [base_cmap(i) for i in range(num_classes)]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, num_classes)

    plt.figure(figsize=(10, 9), dpi=150)
    plt.scatter(X2[:, 0], X2[:, 1], c=y, s=2, cmap=cmap, norm=norm, linewidths=0, alpha=0.9)
    plt.title("t-SNE of Validation Embeddings — Unique Color per Species")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    device = DEVICE
    num_classes = len(val_ds.class_to_idx)
    print(f"[Info] Detected {num_classes} classes in {VAL_DIR}")

    model = FineTuneModel(num_classes=num_classes).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"[Info] Loaded checkpoint from {CKPT_PATH}")

    max_items = SUBSAMPLE if SUBSAMPLE and SUBSAMPLE > 0 else None
    X, y, yhat = extract_embeddings_and_preds(model, val_loader, device, max_items=max_items)
    print(f"[Info] Embeddings shape: {X.shape} (N x D). Running t-SNE...")

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(PERPLEXITY, max(5, len(X)//4 - 1)),
        max_iter=TSNE_ITERS,   # sklearn >= 1.2
        metric=METRIC,
        verbose=1
    )
    X2 = tsne.fit_transform(X)
    
    plot_by_species(X2, y, num_classes, "tsne_val_by_species_unique.png")
    print("plots Saved")

if __name__ == "__main__":
    main()
