from dataclasses import dataclass

@dataclass
class Config:
    data_dir: str = "data/processed"
    img_size: int = 224
    batch_size: int = 16
    num_epochs: int = 10
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 2

    freeze_backbone: bool = True

    model_path: str = "models/best_model.pth"
    log_path: str = "outputs/logs/train_log.csv"
