import os, glob
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

IMG_SIZE   = 224
BATCH_SIZE = 64
BASE_LR    = 3e-4
UNFREEZE_LR= 1e-4
EPOCHS_HEAD= 5
EPOCHS_FULL= 10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR  = "/db/shared/phenotyping/PlantNet/train"
VAL_DIR    = "/db/shared/phenotyping/PlantNet/val"
OUT_CKPT   = "supervised_resnet_finetuned.pth"

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4,0.4,0.4,0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class LabeledDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform
        class_folders = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name:i for i, cls_name in enumerate(class_folders)}
        for cls in class_folders:
            files = glob.glob(os.path.join(root_dir, cls, "*.jpg"))
            self.samples.extend([(f,self.class_to_idx[cls]) for f in files])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path,label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

train_ds = LabeledDataset(TRAIN_DIR, train_tf)
val_ds   = LabeledDataset(VAL_DIR, val_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

NUM_CLASSES = len(train_ds.class_to_idx)

class SupervisedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V2")  
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
    def forward(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc(h)

model = SupervisedResNet(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()

def evaluate(model):
    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, pred = outputs.max(1)
            correct += (pred==labels).sum().item()
            total   += labels.size(0)
    return 100*correct/total

for p in model.encoder.parameters(): 
    p.requires_grad=False

opt = torch.optim.Adam(model.fc.parameters(), lr=BASE_LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_HEAD)

for e in range(EPOCHS_HEAD):
    model.train(); total_loss=0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with autocast():
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total_loss += loss.item()
    sch.step()
    print(f"[Head] Epoch {e+1}/{EPOCHS_HEAD} - Loss {total_loss/len(train_loader):.4f} - Val {evaluate(model):.2f}%")

for p in model.encoder.parameters(): 
    p.requires_grad=True

opt = torch.optim.Adam(model.parameters(), lr=UNFREEZE_LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FULL)

for e in range(EPOCHS_FULL):
    model.train(); total_loss=0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with autocast():
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total_loss += loss.item()
    sch.step()
    print(f"[Full] Epoch {e+1}/{EPOCHS_FULL} - Loss {total_loss/len(train_loader):.4f} - Val {evaluate(model):.2f}%")

torch.save(model.state_dict(), OUT_CKPT)
print(f"Model saved {OUT_CKPT}")
