import numpy as np
import os
import json
import pathlib
import cv2
import torch
import torchvision.transforms as T

file_path = pathlib.Path(__file__).parent.absolute()


def get_transforms(split, img_size):
    """
    Transformaciones para train / val / test trabajando SOLO con tensores.
    Entrada al transform: imagen numpy 48x48 (escala de grises).
    """

    mean, std = 0.5, 0.5

    if split == "train":
        transforms = T.Compose(
            [
                # 1) Numpy (H, W) -> Tensor (1, H, W), valores [0, 1]
                T.ToTensor(),
                # 2) Redimensionar a img_size x img_size
                T.Resize((img_size, img_size)),
                # 3) Data augmentation sobre TENSORES
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.15, 0.15),
                    scale=(0.9, 1.1),
                ),
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0,
                    hue=0,
                ),
                # 4) Normalización
                T.Normalize((mean,), (std,)),
                # 5) Borrado aleatorio (solo tensor)
                T.RandomErasing(p=0.2),
            ]
        )
    else:
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize((img_size, img_size)),
                T.Normalize((mean,), (std,)),
            ]
        )

    deNormalize = UnNormalize(mean=[mean], std=[std])
    return transforms, deNormalize


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        tensor: (C,H,W)
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def to_torch(array: np.ndarray, roll_dims=True):
    """
    Convierte np.ndarray a torch.Tensor.
    """
    if roll_dims:
        if len(array.shape) <= 2:
            array = np.expand_dims(array, axis=2)  # (H, W) -> (H, W, 1)
        array = array.transpose((2, 0, 1))        # (H, W, C) -> (C, H, W)
    tensor = torch.tensor(array)
    return tensor


def to_numpy(tensor: torch.Tensor, roll_dims=True):
    """
    Convierte torch.Tensor a np.ndarray.
    """
    if roll_dims:
        if len(tensor.shape) > 3:
            tensor = tensor.squeeze(0)            # (1, C, H, W) -> (C, H, W)
        tensor = tensor.permute(1, 2, 0)          # (C, H, W) -> (H, W, C)
    array = tensor.detach().cpu().numpy()
    return array


def add_img_text(img: np.ndarray, text_label: str):
    """
    Agrega texto a una imagen (numpy BGR o GRAY).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 0, 0)
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(text_label, font, fontScale, thickness)

    x1, y1 = 0, text_h
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), (255, 255, 255), -1)
    if img.ndim == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.putText(img, text_label, (x1, y1), font, fontScale, fontColor, thickness)
    return img


def create_train_val_split():
    """
    Crea split.json para train / val usando train.csv
    """
    import pandas as pd

    train_csv = file_path / "data/train.csv"
    df = pd.read_csv(train_csv)
    n_samples = len(df)

    val_samples = np.random.choice(n_samples, size=int(n_samples // 5), replace=False)
    train_samples = np.setdiff1d(np.arange(n_samples), val_samples)

    sample_dct = {"train": train_samples.tolist(), "val": val_samples.tolist()}
    outfile = file_path / "data/split.json"
    with open(outfile, "w") as f:
        json.dump(sample_dct, f, indent=2)
