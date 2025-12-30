import os
import csv
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from config import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "val"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if cfg.freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    with open(cfg.log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_sec"])

    best_val_acc = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        start = time.time()

        model.train()
        train_loss, train_acc = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(accuracy(out, y))

        model.eval()
        val_loss, val_acc = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)

                val_loss.append(loss.item())
                val_acc.append(accuracy(out, y))

        tl, ta = np.mean(train_loss), np.mean(train_acc)
        vl, va = np.mean(val_loss), np.mean(val_acc)
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{cfg.num_epochs} | "
              f"Train Acc: {ta:.3f} | Val Acc: {va:.3f}")

        with open(cfg.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tl, ta, vl, va, round(elapsed, 2)])

        if va > best_val_acc:
            best_val_acc = va
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes
            }, cfg.model_path)

    print("🎉 Eğitim tamamlandı. En iyi Val Acc:", best_val_acc)


if __name__ == "__main__":
    main()
