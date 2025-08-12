# simclr_pretrain.py
import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 14
TEMPERATURE = 0.5
LR = 3e-4
DATA_DIR = '/db/shared/phenotyping/PlantNet/train'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.files = glob.glob(os.path.join(root_dir, '**/*.jpg'), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        xi = self.transform(img)
        xj = self.transform(img)
        return xi, xj

class SimCLRModel(nn.Module):
    def __init__(self, base_model='resnet50', projection_dim=128):
        super().__init__()
        resnet = models.__dict__[base_model](weights=None)
        # Encoder without final classification head
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features
        self.projector = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        z = self.projector(h)
        return F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature

    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels, labels], dim=0)

    loss_fn = nn.CrossEntropyLoss()
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)

    positives = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)], dim=0).to(z.device)
    logits = sim
    targets = positives
    return loss_fn(logits, targets)

def train_simclr():
    print("Preparing dataset...", flush=True)
    dataset = UnlabeledDataset(DATA_DIR, simclr_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = SimCLRModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch_idx, (x_i, x_j) in enumerate(loader):
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            z_i = model(x_i)
            z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j, TEMPERATURE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}", flush=True)
        torch.cuda.empty_cache()

    torch.save(model.encoder.state_dict(), "simclr_encoder.pth")
    torch.save(model.state_dict(), "simclr_full.pth")

if __name__ == '__main__':
    train_simclr()
