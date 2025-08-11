import os
import glob
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 14
TEMPERATURE = 0.5
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = '/db/shared/phenotyping/PlantNet'
PRETRAINED_PATH = "simclr_pretrained_encoder.pth"

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze()
        z = self.projector(h)
        return F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim = sim / temperature

    labels = torch.cat([torch.arange(N) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim = sim[~mask].view(sim.shape[0], -1)

    positives = sim[labels.bool()].view(sim.shape[0], -1)
    negatives = sim[~labels.bool()].view(sim.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)
    return F.cross_entropy(logits, labels)

def pretrain_simclr():
    print("Loading unlabeled dataset for SimCLR...")
    dataset = UnlabeledDataset(os.path.join(DATA_DIR, 'train'), simclr_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = SimCLRModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x_i, x_j in loader:
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            z_i = model(x_i)
            z_j = model(x_j)
            loss = nt_xent_loss(z_i, z_j, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Pretrain] Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")
        torch.cuda.empty_cache()

    torch.save(model.encoder.state_dict(), PRETRAINED_PATH)
    print(f"Pretrained encoder saved to {PRETRAINED_PATH}")

def train_finetune():
    print("Loading labeled dataset for fine-tuning...")
    dataset_train = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transform)
    dataset_test = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'test'), transform=test_transform)
    num_classes = len(dataset_train.classes)

    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet50(pretrained=False)
    state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, 15):
        # Train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        print(f"[Finetune] Epoch {epoch} - Test Loss: {test_loss:.4f} - Accuracy: {acc:.2f}%")
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, choices=["pretrain", "finetune"], required=True)
    args = parser.parse_args()

    if args.phase == "pretrain":
        pretrain_simclr()
    elif args.phase == "finetune":
        train_finetune()
