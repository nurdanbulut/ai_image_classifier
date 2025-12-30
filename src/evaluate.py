import torch
import numpy as np
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Ayarlar ----
DATA_DIR = "data/processed/test"
MODEL_PATH = "models/best_model.pth"
IMG_SIZE = 224
BATCH_SIZE = 16

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Transform (test = augmentation yok)
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_ds = datasets.ImageFolder(DATA_DIR, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(test_ds.classes)
    print("Classes:", test_ds.classes)

    # Model
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    # ---- METRİKLER ----
    report = classification_report(
        y_true, y_pred,
        target_names=test_ds.classes,
        digits=4
    )
    print("\nClassification Report:\n")
    print(report)

    # ---- RAPOR DOSYASINA KAYDET ----
    os.makedirs("outputs/reports", exist_ok=True)
    with open("outputs/reports/test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # ---- CONFUSION MATRIX ----
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=test_ds.classes,
                yticklabels=test_ds.classes,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()

    plt.savefig("outputs/figures/confusion_matrix.png")
    plt.show()

    print("✅ Test değerlendirmesi tamamlandı.")
    print("📄 Rapor: outputs/reports/test_classification_report.txt")
    print("📊 Confusion matrix: outputs/figures/confusion_matrix.png")

if __name__ == "__main__":
    main()
