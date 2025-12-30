import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = "outputs/logs/train_log.csv"
OUT_DIR = "outputs/figures"

def main():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Log bulunamadı: {LOG_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(LOG_PATH)

    # Beklenen kolonlar:
    # epoch, train_loss, train_acc, val_loss, val_acc, time_sec

    # 1) Accuracy grafiği
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_curve.png"), dpi=200)
    plt.show()

    # 2) Loss grafiği
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=200)
    plt.show()

    print("✅ Grafikler kaydedildi:")
    print(" - outputs/figures/accuracy_curve.png")
    print(" - outputs/figures/loss_curve.png")

if __name__ == "__main__":
    main()
