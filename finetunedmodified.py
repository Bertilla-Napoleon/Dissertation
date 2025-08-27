import os
import glob
import json
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

IMG_SIZE = 224
BATCH_SIZE = 64
BASE_LR = 3e-4        
FULL_LR = 1e-4        
WEIGHT_DECAY = 1e-4
EPOCHS_HEAD = 5
EPOCHS_FULL = 10
NUM_CLASSES = 1081
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "/db/shared/phenotyping/PlantNet/train"
VAL_DIR   = "/db/shared/phenotyping/PlantNet/val"
ENCODER_CKPT = "simclr_pretrained_encoder.pth"
FINAL_CKPT   = "simclr_finetuned_model.pth"

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(42)

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

train_dataset = LabeledDataset(TRAIN_DIR, train_transform)
val_dataset   = LabeledDataset(VAL_DIR,   val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

class FineTuneModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        h = self.encoder(x).flatten(1)
        return self.fc(h)

model = FineTuneModel(num_classes=NUM_CLASSES).to(DEVICE)
model = model.to(memory_format=torch.channels_last)

pretrained_dict = torch.load(ENCODER_CKPT, map_location=DEVICE)
model.encoder.load_state_dict(pretrained_dict, strict=True)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()

def evaluate(model):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (total_loss / max(1, total), 100.0 * correct / max(1, total))

TRAIN_LOSS = []
VAL_LOSS   = []

for p in model.encoder.parameters():
    p.requires_grad = False

optim_head = torch.optim.AdamW(model.fc.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(optim_head, T_max=EPOCHS_HEAD)

for epoch in range(EPOCHS_HEAD):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optim_head.zero_grad(set_to_none=True)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optim_head)
        scaler.update()
        total_loss += loss.item()

    sched_head.step()
    val_loss, _ = evaluate(model) 
    epoch_train_loss = total_loss / max(1, len(train_loader))
    TRAIN_LOSS.append(float(epoch_train_loss))
    VAL_LOSS.append(float(val_loss))
    print(f"[Head Train] Epoch {epoch+1}/{EPOCHS_HEAD} - TrainLoss: {epoch_train_loss:.4f} - ValLoss: {val_loss:.4f}")

for p in model.encoder.parameters():
    p.requires_grad = True

optim_full = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": FULL_LR * 0.1, "weight_decay": WEIGHT_DECAY},
    {"params": model.fc.parameters(),      "lr": FULL_LR,       "weight_decay": WEIGHT_DECAY},
])
sched_full = torch.optim.lr_scheduler.CosineAnnealingLR(optim_full, T_max=EPOCHS_FULL)

for epoch in range(EPOCHS_FULL):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optim_full.zero_grad(set_to_none=True)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optim_full)
        scaler.update()
        total_loss += loss.item()

    sched_full.step()
    val_loss, _ = evaluate(model)  
    epoch_train_loss = total_loss / max(1, len(train_loader))
    TRAIN_LOSS.append(float(epoch_train_loss))
    VAL_LOSS.append(float(val_loss))
    print(f"[Full Fine-tune] Epoch {epoch+1}/{EPOCHS_FULL} - TrainLoss: {epoch_train_loss:.4f} - ValLoss: {val_loss:.4f}")

torch.save(model.state_dict(), FINAL_CKPT)
print(f"Saved model to {FINAL_CKPT}")

with open("finetune_loss_history.json", "w") as f:
    json.dump({"train_loss": TRAIN_LOSS, "val_loss": VAL_LOSS}, f)

plt.figure()
plt.plot(range(1, len(TRAIN_LOSS)+1), TRAIN_LOSS, label="Train loss")
plt.plot(range(1, len(VAL_LOSS)+1),   VAL_LOSS,   label="Val loss")
plt.xlabel("Epoch (Head + Full)")
plt.ylabel("Loss")
plt.title("Fine-tuning Loss Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("finetune_loss_curve.png", dpi=150)
print("Saved: finetune_loss_history.json, finetune_loss_curve.png")
