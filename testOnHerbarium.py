import os, glob
from collections import OrderedDict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 8

HERBARIUM_DIR = "/db/shared/phenotyping/Herbarium"         
PLANTNET_TRAIN_DIR = "/db/shared/phenotyping/PlantNet/train" 
SIMCLR_ENCODER_WEIGHTS = "simclr_pretrained_encoder.pth"

KNN_K = 3                 
MAX_WILD_KEEP = 15000     

PCA_DIM = 50
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_PLOT_PATH = "tsne_plot.png" 


eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


class FlatImageDataset(Dataset):
    def __init__(self, filepaths, transform):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img


def list_herbarium_pngs(root_dir):
    return sorted(glob.glob(os.path.join(root_dir, "*.png")))


def list_all_wild_jpegs(root_dir):
    jpgs = []
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        if os.path.isdir(cls_dir):
            jpgs.extend(glob.glob(os.path.join(cls_dir, "*.jpg")))
    return sorted(jpgs)


class SimCLREncoder(nn.Module):
    def __init__(self, base_model="resnet50"):
        super().__init__()
        resnet = models.__dict__[base_model](weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  
    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x).flatten(1)  
        return feats


def load_pretrained_encoder(weights_path):
    model = SimCLREncoder().to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    model.encoder.load_state_dict(state, strict=True)
    model.eval()
    return model


def extract_embeddings(model, filepaths):
    ds = FlatImageDataset(filepaths, eval_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    feats = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            feats.append(model(imgs).cpu().numpy())
    return np.concatenate(feats, axis=0)  


def l2_normalize(arr):
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norm


def select_relevant_wild_via_knn(herb_feats, wild_feats, wild_paths, k=KNN_K, max_keep=MAX_WILD_KEEP):
    herb_norm = l2_normalize(herb_feats)
    wild_norm = l2_normalize(wild_feats)

    nn = NearestNeighbors(n_neighbors=min(k, wild_norm.shape[0]),
                          metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(wild_norm)

    _, nbr_idx = nn.kneighbors(herb_norm, return_distance=True) 

    flat_idxs = nbr_idx.reshape(-1).tolist()
    uniq = list(OrderedDict.fromkeys(flat_idxs).keys())

    if len(uniq) > max_keep:
        uniq = uniq[:max_keep]

    sel_wild_paths = [wild_paths[i] for i in uniq]
    sel_wild_feats = wild_feats[uniq]

    return sel_wild_feats, sel_wild_paths


def run_tsne(embeddings):
    pca = PCA(n_components=min(PCA_DIM, embeddings.shape[1]), random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                n_iter=TSNE_N_ITER, init="pca", learning_rate="auto",
                random_state=42, verbose=1)
    return tsne.fit_transform(emb_pca)


def silhouette_kmeans(X, n_clusters):
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    return silhouette_score(X, labels, metric="euclidean")


def save_tsne_plot(emb2d, domain_labels, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    mask_wild = (domain_labels == 0)
    mask_herb = (domain_labels == 1)
    plt.scatter(emb2d[mask_wild, 0], emb2d[mask_wild, 1], s=4, alpha=0.6, label="Wild (k-NN subset)")
    plt.scatter(emb2d[mask_herb, 0], emb2d[mask_herb, 1], s=4, alpha=0.6, label="Herbarium (all)")
    plt.legend(loc="best")
    plt.title("t-SNE of SimCLR Embeddings: Herbarium + k-NN Wild Neighbors")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] t-SNE plot → {save_path}")

def main():
    print(f"Device: {DEVICE}")

    herb_paths = list_herbarium_pngs(HERBARIUM_DIR)
    wild_paths = list_all_wild_jpegs(PLANTNET_TRAIN_DIR)
    if len(herb_paths) == 0:
        raise RuntimeError(f"No .png in {HERBARIUM_DIR}")
    if len(wild_paths) == 0:
        raise RuntimeError(f"No .jpg in {PLANTNET_TRAIN_DIR}")
    print(f"Herbarium: {len(herb_paths):,} | Wild: {len(wild_paths):,}")

    print("Loading pretrained SimCLR encoder...")
    encoder = load_pretrained_encoder(SIMCLR_ENCODER_WEIGHTS)

    print("Extracting herbarium embeddings (all)...")
    herb_feats = extract_embeddings(encoder, herb_paths)

    print("Extracting wild embeddings (all)...")
    wild_feats = extract_embeddings(encoder, wild_paths)

    print(f"Selecting wild neighbors via k-NN (k={KNN_K})...")
    sel_wild_feats, sel_wild_paths = select_relevant_wild_via_knn(
        herb_feats, wild_feats, wild_paths, k=KNN_K, max_keep=MAX_WILD_KEEP
    )
    print(f"Selected wild neighbors: {len(sel_wild_paths):,}")

    all_feats = np.vstack([sel_wild_feats, herb_feats])
    all_domain = np.hstack([
        np.zeros(len(sel_wild_feats), dtype=np.int32),  # 0=wild
        np.ones(len(herb_feats), dtype=np.int32)        # 1=herbarium
    ])

    print("Running PCA → t-SNE (this can be heavy for large N)...")
    emb2d = run_tsne(all_feats)
    save_tsne_plot(emb2d, all_domain, TSNE_PLOT_PATH)

    print("Computing silhouette scores...")
    sil_herb = silhouette_kmeans(herb_feats, n_clusters=100)
    sil_comb = silhouette_kmeans(all_feats, n_clusters=150)
    domain_sil = silhouette_score(all_feats, all_domain, metric="euclidean")

    print("\n================ Silhouette Results ================")
    print(f"Herbarium-only (KMeans, K=100): {sil_herb:.4f}")
    print(f"Combined (KMeans, K=150):      {sil_comb:.4f}")
    print(f"Domain silhouette (wild vs herb): {domain_sil:.4f}")

if __name__ == "__main__":
    main()
