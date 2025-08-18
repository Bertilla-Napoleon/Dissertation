import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

IMG_SIZE = 224
BATCH_SIZE = 64
BASE_LR = 3e-4
UNFREEZE_LR = 1e-4
EPOCHS_HEAD = 5      
EPOCHS_FULL = 10     
NUM_CLASSES = 1081
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TSNE_SUBSAMPLE = None   
TSNE_PERPLEXITY = 30     
TSNE_ITERS = 1000

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class LabeledDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform
        class_folders = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_folders)}
        for cls in class_folders:
            files = glob.glob(os.path.join(root_dir, cls, "*.jpg"))
            self.samples.extend([(f, self.class_to_idx[cls]) for f in files])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label

train_dataset = LabeledDataset("/db/shared/phenotyping/PlantNet/train", train_transform)
val_dataset   = LabeledDataset("/db/shared/phenotyping/PlantNet/val",  val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

class FineTuneModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])   
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        h = self.encoder(x).flatten(1)  # safer than squeeze()
        out = self.fc(h)
        return out

model = FineTuneModel(num_classes=NUM_CLASSES).to(DEVICE)

pretrained_dict = torch.load("simclr_pretrained_encoder.pth", map_location=DEVICE)
model.encoder.load_state_dict(pretrained_dict, strict=True)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.fc.parameters(), lr=BASE_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD)

for epoch in range(EPOCHS_HEAD):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    scheduler.step()
    val_acc = evaluate(model)
    print(f"[Head Train] Epoch {epoch+1}/{EPOCHS_HEAD} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%", flush=True)

for param in model.encoder.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=UNFREEZE_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)

for epoch in range(EPOCHS_FULL):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    scheduler.step()
    val_acc = evaluate(model)
    print(f"[Full Fine-tune] Epoch {epoch+1}/{EPOCHS_FULL} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%", flush=True)

torch.save(model.state_dict(), "simclr_finetuned_model.pth")

@torch.no_grad()
def extract_embeddings_and_preds(model, loader, max_items=None):
    model.eval()
    feats, labels, preds = [], [], []
    for imgs, y in loader:
        imgs = imgs.to(DEVICE)
        y = y.to(DEVICE)
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

print("Extracting validation embeddings for t-SNE...")
max_items = TSNE_SUBSAMPLE if TSNE_SUBSAMPLE is not None else None
X, y, yhat = extract_embeddings_and_preds(model, val_loader, max_items=max_items)
print(f"Embeddings shape: {X.shape} (N x D). Running t-SNE...", flush=True)

effective_perplexity = min(TSNE_PERPLEXITY, max(5, len(X)//4 - 1))
tsne = TSNE(
    n_components=2,
    init="pca",
    learning_rate="auto",
    perplexity=effective_perplexity,
    n_iter=TSNE_ITERS,
    metric="cosine",
    verbose=1
)
X2 = tsne.fit_transform(X)

plt.figure(figsize=(9, 8), dpi=140)
plt.scatter(X2[:, 0], X2[:, 1], c=(y % 20), s=2, alpha=0.6)
plt.title("t-SNE of Validation Embeddings (colored by class mod 20)")
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.savefig("tsne_val_by_class.png")

correct = (y == yhat).astype(np.int32)
plt.figure(figsize=(9, 8), dpi=140)
plt.scatter(X2[correct==1, 0], X2[correct==1, 1], s=3, alpha=0.6, label="Correct")
plt.scatter(X2[correct==0, 0], X2[correct==0, 1], s=3, alpha=0.6, label="Incorrect")
plt.title("t-SNE of Validation Embeddings (Correct vs Incorrect)")
plt.legend(markerscale=4)
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.savefig("tsne_val_correct_incorrect.png")
