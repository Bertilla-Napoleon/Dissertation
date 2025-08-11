import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import os

IMG_SIZE = 224
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 10
NUM_CLASSES = 1081  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

finetune_transform = transforms.Compose([
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

train_dataset = LabeledDataset("/db/shared/phenotyping/PlantNet/train", finetune_transform)
val_dataset = LabeledDataset("/db/shared/phenotyping/PlantNet/val", finetune_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

class FineTuneModel(nn.Module):
    def __init__(self, projection_dim=128, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove fc
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        with torch.no_grad():  # freeze encoder in first stage
            h = self.encoder(x).squeeze()
        out = self.fc(h)
        return out

model = FineTuneModel(num_classes=NUM_CLASSES).to(DEVICE)

pretrained_dict = torch.load("simclr_pretrained_encoder.pth", map_location=DEVICE)
model.encoder.load_state_dict(pretrained_dict)

for param in model.encoder.parameters():
    param.requires_grad = True  # set False if you want frozen start

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Val Acc: {acc:.2f}%")

torch.save(model.state_dict(), "simclr_finetuned_model.pth")
