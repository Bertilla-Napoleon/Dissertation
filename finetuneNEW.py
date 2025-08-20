import os, glob, math, copy
from collections import Counter
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

TRAIN_DIR  = "/db/shared/phenotyping/PlantNet/train"
VAL_DIR    = "/db/shared/phenotyping/PlantNet/val"
SIMCLR_CKPT = "simclr_pretrained_encoder.pth"
OUT_CKPT    = "simclr_finetuned_supcon.pth"
OUT_CKPT_EMA= "simclr_finetuned_supcon_ema.pth"

IMG_SIZE    = 224
BATCH_SIZE  = 64
EPOCHS_HEAD = 5          
EPOCHS_FULL = 40         
BASE_LR     = 3e-4       
UNFREEZE_LR = 1e-4       
WD_HEAD     = 1e-4       
WD_FULL     = 2e-4       
WARMUP_EPOCHS = 2        
SUPCON_LAMBDA = 0.2      
SUPCON_TAU    = 0.07     
USE_EMA       = True
EMA_DECAY     = 0.999

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIXED_PRECISION = True
NUM_WORKERS = 4
PIN_MEMORY  = True
GRAD_CLIP_NORM = 1.0

class LabeledDataset(Dataset):
    def __init__(self, root_dir: str, transform):
        self.samples = []
        self.transform = transform
        class_folders = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
        if not class_folders:
            raise RuntimeError(f"No class folders in {root_dir}")
        self.class_to_idx = {c:i for i,c in enumerate(class_folders)}
        for c in class_folders:
            files = glob.glob(os.path.join(root_dir, c, "*.jpg"))
            self.samples.extend([(f, self.class_to_idx[c]) for f in files])
        if not self.samples:
            raise RuntimeError(f"No images found under {root_dir}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_ds = LabeledDataset(TRAIN_DIR, train_tf)
val_ds   = LabeledDataset(VAL_DIR,   val_tf)

NUM_CLASSES = len(train_ds.class_to_idx)
print(f"[Info] Detected {NUM_CLASSES} classes.")

counts = Counter([lbl for _, lbl in train_ds.samples])
weights = torch.tensor([1.0 / counts[lbl] for _, lbl in train_ds.samples], dtype=torch.float)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

class FineTuneModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x, return_features: bool=False):
        h = self.encoder(x).flatten(1)  
        logits = self.fc(h)             
        if return_features:
            return logits, h
        return logits

model = FineTuneModel(NUM_CLASSES).to(DEVICE)

enc_state = torch.load(SIMCLR_CKPT, map_location=DEVICE)
model.encoder.load_state_dict(enc_state, strict=True)
print(f"[Info] Loaded SimCLR encoder from {SIMCLR_CKPT}")

if USE_EMA:
    ema_model = copy.deepcopy(model).to(DEVICE)
    for p in ema_model.parameters(): p.requires_grad = False
else:
    ema_model = None

criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float=0.07) -> torch.Tensor:
    z = F.normalize(features, dim=1)
    sim = torch.matmul(z, z.T) / temperature                
    B = sim.size(0)
    mask = torch.eq(labels.view(-1,1), labels.view(1,-1)).float().to(features.device)
    logits_mask = (1 - torch.eye(B, device=features.device))
    sim = sim - 1e9 * (1 - logits_mask)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)
    return -mean_log_prob_pos.mean()

@torch.no_grad()
def evaluate(m: nn.Module) -> float:
    m.eval()
    correct = total = 0
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = m(imgs)
        preds = logits.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return 100.0 * correct / max(1, total)

def update_ema(model: nn.Module, ema_model: nn.Module, decay: float):
    if ema_model is None: return
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

def total_steps(num_epochs: int) -> int:
    return num_epochs * len(train_loader)

def lr_schedule(step: int, total: int, warmup: int):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

def run_train(head_only: bool):
    global_step = 0
    epochs = EPOCHS_HEAD if head_only else EPOCHS_FULL

    for p in model.encoder.parameters(): p.requires_grad = (not head_only)

    if head_only:
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=BASE_LR, weight_decay=WD_HEAD)
        base_lrs = [BASE_LR] 
    else:
        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': UNFREEZE_LR * 0.5},
            {'params': model.fc.parameters(),      'lr': UNFREEZE_LR * 3.0},
        ], weight_decay=WD_FULL)
        base_lrs = [UNFREEZE_LR * 0.5, UNFREEZE_LR * 3.0]

    total = total_steps(epochs)
    warmup = WARMUP_EPOCHS * len(train_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            scale = lr_schedule(global_step, total, warmup)
            for pg, base in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = base * scale

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                logits, feats = model(imgs, return_features=True)
                ce = criterion_ce(logits, labels)
                con = supcon_loss(feats, labels, SUPCON_TAU)
                loss = ce + (0.0 if head_only else SUPCON_LAMBDA * con)

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            update_ema(model, ema_model, EMA_DECAY)

            running += float(loss.item())
            global_step += 1

        val_acc = evaluate(ema_model if USE_EMA else model)
        print(f"[{'Head' if head_only else 'Full'}] Epoch {epoch}/{epochs} - "
              f"Loss: {running/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")

run_train(head_only=True)
run_train(head_only=False)

torch.save(model.state_dict(), OUT_CKPT)
print(f"{OUT_CKPT} Saved")
if USE_EMA and ema_model is not None:
    torch.save(ema_model.state_dict(), OUT_CKPT_EMA)
    print(f"{OUT_CKPT_EMA} Saved")
