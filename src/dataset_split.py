import random
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def split_and_copy(class_dir: Path, class_name: str):
    images = list_images(class_dir)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        dst_dir = OUT_DIR / split_name / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_imgs:
            shutil.copy2(img_path, dst_dir / img_path.name)

    return n, len(train_imgs), len(val_imgs), len(test_imgs)

def main():
    random.seed(SEED)

    if not RAW_DIR.exists():
        raise RuntimeError("data/raw bulunamadı. Dataset'i data/raw içine koymalısın.")

    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if len(class_dirs) == 0:
        raise RuntimeError("data/raw içinde sınıf klasörleri yok. (10 klasör olmalı)")

    # Çıkış klasörünü sıfırla
    ensure_empty_dir(OUT_DIR)

    summary = []
    for cdir in sorted(class_dirs):
        total, tr, va, te = split_and_copy(cdir, cdir.name)
        summary.append((cdir.name, total, tr, va, te))

    # Rapor tablosunu kaydet
    report_path = Path("outputs/reports/dataset_split_summary.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("class\ttotal\ttrain\tval\ttest\n")
        for row in summary:
            f.write("\t".join(map(str, row)) + "\n")

    print("✅ Split tamamlandı!")
    print("Sınıf sayısı:", len(summary))
    print("Rapor:", report_path)

if __name__ == "__main__":
    main()
