"""PyTorch Training Script for Vial Slot Defect Classification.

Supports training on:
- Synthetic crops rendered from the Isaac Sim digital twin
- Labeled crops from review sessions
"""

import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from .rules import CLASSES


class VialCropDataset(Dataset):
    """Dataset of cropped vial positions with class labels."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        # Supports either directory-per-class or json manifest
        manifest_path = self.data_dir / "labels.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for res in data.get('results', []):
                for slot in res.get('slots', []):
                    c = slot.get('class')
                    crop_path = self.data_dir / slot.get('crop_path', '')
                    if c in CLASSES and crop_path.exists():
                        self.samples.append((str(crop_path), CLASSES.index(c)))
        else:
            for idx, cls_name in enumerate(CLASSES):
                cls_folder = self.data_dir / cls_name
                if cls_folder.exists():
                    for img_p in cls_folder.glob("*.jpg"):
                        self.samples.append((str(img_p), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(num_classes=5, pretrained=True):
    """Builds a lightweight MobileNetV3 model for real-time inference."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train(data_dir, output_model_path="models/vial_mobilenet.pth", epochs=15, batch_size=32, lr=3e-4):
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = VialCropDataset(data_dir, transform=transform_train)
    if len(dataset) == 0:
        print(f"[!] No valid samples found in {data_dir}. Ensure data folder exists.")
        return

    # Compute balanced sample weights to prevent majority class collapse
    import numpy as np
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets, minlength=len(CLASSES))
    print(f"[*] Dataset class distribution: {dict(zip(CLASSES, class_counts))}")
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[t] for t in targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False

    model = build_model(num_classes=len(CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[*] Starting balanced training on {len(dataset)} samples using {device}...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        acc = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} - Loss: {avg_loss:.4f} - Accuracy: {acc*100:.2f}%")

    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"[+] Saved model weights to {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to cropped dataset or review folder")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", default="models/vial_mobilenet.pth")
    args = parser.parse_args()
    train(args.data_dir, output_model_path=args.output, epochs=args.epochs)
