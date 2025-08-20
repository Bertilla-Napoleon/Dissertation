import os, glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

VAL_DIR     = "/db/shared/phenotyping/PlantNet/val"
CKPT_PATH   = "simclr_finetuned_model.pth"
IMG_SIZE    = 224
BATCH_SIZE  = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUBSAMPLE   = None      
MAX_CLASSES = 10        
USE_STANDARDIZE = True  

OUT_PREFIX  = "pca_val" 

class LabeledDataset(Dataset):
    def __init__(self, root_dir, transform, max_classes=None):
        self.samples = []
        self.transform = transform
        class_folders = sorted(d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d)))
        if len(class_folders) == 0:
            raise RuntimeError(f"No class folders found in {root_dir}")
        if max_classes is not None:
            class_folders = class_folders[:max_classes]

        self.class_to_idx = {c: i for i, c in enumerate(class_folders)}
        for c in class_folders:
            files = glob.glob(os.path.join(root_dir, c, "*.jpg"))
            self.samples.extend([(f, self.class_to_idx[c]) for f in files])

        if len(self.samples) == 0:
            raise RuntimeError("No images found for the selected classes.")

    def __len__(self): return len(self.samples)

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
val_ds = LabeledDataset(VAL_DIR, val_tf, max_classes=MAX_CLASSES)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

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
def extract_embeddings(model, loader, device, max_items=None):
    model.eval()
    feats, labels = [], []
    for imgs, y in loader:
        imgs = imgs.to(device)
        h = model.encoder(imgs).flatten(1)   
        feats.append(h.cpu())
        labels.append(y)
        if max_items is not None and sum(len(a) for a in labels) >= max_items:
            break
    X = torch.cat(feats, 0).numpy()
    y = torch.cat(labels, 0).numpy()
    if max_items is not None and len(y) > max_items:
        X = X[:max_items]; y = y[:max_items]
    return X, y

def make_palette(k):
    if k <= 10:
        base = cm.get_cmap("tab10", k)
    else:
        base = cm.get_cmap("nipy_spectral", k)
    return ListedColormap([base(i) for i in range(k)])

def remap_labels(y):
    uniq = np.unique(y)
    m = {old:i for i, old in enumerate(uniq)}
    y_new = np.array([m[int(t)] for t in y])
    return y_new, uniq

def idx_to_names(ds):
    return [c for c, _ in sorted(ds.class_to_idx.items(), key=lambda x: x[1])]

def plot_pca_2d(Z2, y_plot, names, out_png):
    k = len(names)
    cmap = make_palette(k)
    plt.figure(figsize=(10, 9), dpi=150)
    sc = plt.scatter(Z2[:,0], Z2[:,1], c=y_plot, s=10, alpha=0.98, cmap=cmap, linewidths=0)
    plt.title(f"PCA (2D) of Validation Embeddings — {k} classes, N={len(y_plot)}")
    plt.xticks([]); plt.yticks([])
    handles, _ = sc.legend_elements(num=k)
    plt.legend(handles, names, loc="best", fontsize=8, markerscale=2, frameon=True, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

def plot_pca_3d(Z3, y_plot, names, out_png):
    k = len(names)
    cmap = make_palette(k)
    fig = plt.figure(figsize=(10, 9), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], c=y_plot, s=10, alpha=0.98, cmap=cmap, linewidths=0)
    ax.set_title(f"PCA (3D) of Validation Embeddings — {k} classes, N={len(y_plot)}")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticks([])
    handles, _ = sc.legend_elements(num=k)
    ax.legend(handles, names, loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

def main():
    device = DEVICE
    # Load checkpoint and infer trained head size
    state = torch.load(CKPT_PATH, map_location=device)
    fc_w = next(v for k, v in state.items() if k.endswith("fc.weight"))
    num_trained_classes = fc_w.shape[0]
    model = FineTuneModel(num_classes=num_trained_classes).to(device)
    model.load_state_dict(state, strict=True)

    max_items = SUBSAMPLE if SUBSAMPLE and SUBSAMPLE > 0 else None
    X, y = extract_embeddings(model, val_loader, device, max_items=max_items)
    print(f"[Info] Embeddings: {X.shape}. Running PCA...")

    pca2 = PCA(n_components=2, random_state=0)
    Z2 = pca2.fit_transform(Xproc)
    print(f"[PCA-2D] Explained variance ratio: {pca2.explained_variance_ratio_.sum():.3f} "
          f"({pca2.explained_variance_ratio_[0]:.3f}, {pca2.explained_variance_ratio_[1]:.3f})")

    pca3 = PCA(n_components=3, random_state=0)
    Z3 = pca3.fit_transform(Xproc)
    print(f"[PCA-3D] Explained variance ratio: {pca3.explained_variance_ratio_.sum():.3f}")

    y_plot, uniq = remap_labels(y)
    names = idx_to_names(val_ds)

    plot_pca_2d(Z2, y_plot, names, f"{OUT_PREFIX}_2d.png")
    plot_pca_3d(Z3, y_plot, names, f"{OUT_PREFIX}_3d.png")
    print("Saved PCA plots")

if __name__ == "__main__":
    main()
