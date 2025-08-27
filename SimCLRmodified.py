import os
import glob
import math
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

IMG_SIZE     = 224
BATCH_SIZE   = 64          
EPOCHS       = 14          
TEMPERATURE  = 0.5
BASE_LR      = 3e-4
WARMUP_EPOCHS= 2           
WEIGHT_DECAY = 1e-4
DATA_DIR     = '/db/shared/phenotyping/PlantNet/train'
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_CKPT = "simclr_pretrained_encoder_final.pth"
FULL_CKPT    = "simclr_full_final.pth"

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(42)

EPOCH_LOSSES = []

def _gaussian_kernel(img_size):
    k = int(0.1 * img_size)
    k = max(3, k | 1)  
    return k

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), 
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=_gaussian_kernel(IMG_SIZE), sigma=(0.1, 2.0)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform, exts=(".jpg",".jpeg",".png",".JPG",".JPEG",".PNG")):
        self.files = []
        for ext in exts:
            self.files.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
        if not self.files:
            raise RuntimeError(f"No images found under {root_dir}")
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        xi = self.transform(img)
        xj = self.transform(img)
        return xi, xj

class ProjectorMLP(nn.Module):
    def __init__(self, in_dim, hidden=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )
    def forward(self, x): return self.net(x)

class SimCLRModel(nn.Module):
    def __init__(self, base_model='resnet50', projection_dim=128):
        super().__init__()
        resnet = models.__dict__[base_model](weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) 
        self.in_features = resnet.fc.in_features
        self.projector = ProjectorMLP(self.in_features, hidden=2048, out_dim=projection_dim)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)   
        h = self.encoder(x).flatten(1)                
        z = self.projector(h)
        return F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)
    positives = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    targets = positives
    return F.cross_entropy(sim, targets)

def cosine_with_warmup(epoch, base_lr, epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    e = epoch - warmup_epochs
    E = max(1, epochs - warmup_epochs)
    return 0.5 * base_lr * (1 + math.cos(math.pi * e / E))

def train_simclr():
    dataset = UnlabeledDataset(DATA_DIR, simclr_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    model = SimCLRModel(base_model='resnet50', projection_dim=128).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        lr = cosine_with_warmup(epoch, BASE_LR, EPOCHS, WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for x_i, x_j in loader:
            x_i = x_i.to(DEVICE, non_blocking=True)
            x_j = x_j.to(DEVICE, non_blocking=True)

            with autocast():
                z_i = model(x_i)
                z_j = model(x_j)
                loss = nt_xent_loss(z_i, z_j, TEMPERATURE)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch [{epoch+1}/{EPOCHS}] - LR: {lr:.2e} - Average Loss: {avg_loss:.4f}", flush=True)
        EPOCH_LOSSES.append(float(avg_loss))
        torch.cuda.empty_cache()

    with open("simclr_pretrain_losses.json", "w") as f:
        json.dump({"epoch_losses": EPOCH_LOSSES}, f)

    plt.figure()
    plt.plot(range(1, len(EPOCH_LOSSES)+1), EPOCH_LOSSES, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss (avg)")
    plt.title("SimCLR Pretraining Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("simclr_pretrain_loss.png", dpi=150)

    torch.save(model.encoder.state_dict(), ENCODER_CKPT)
    torch.save(model.state_dict(), FULL_CKPT)
    print(f"Model and loss curve Saved")

if __name__ == '__main__':
    train_simclr()
