"""
Dataset FER2013 para expresiones faciales.
"""

import pathlib
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from .utils import (
    to_numpy,
    to_torch,
    add_img_text,
    get_transforms,
)
import json

EMOTIONS_MAP = {
    0: "Enojo",
    1: "Disgusto",
    2: "Miedo",
    3: "Alegria",
    4: "Tristeza",
    5: "Sorpresa",
    6: "Neutral",
}

file_path = pathlib.Path(__file__).parent.absolute()


def get_loader(split, batch_size, shuffle=True, num_workers=0):
    """
    Regresa dataset y dataloader para train / val / test.
    """
    dataset = FER2013(root=file_path, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # CPU only
    )
    return dataset, dataloader


class FER2013(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.img_size = 48
        self.target_transform = target_transform
        self.split = split
        self.root = root
        self.unnormalize = None

        self.transform, self.unnormalize = get_transforms(
            split=self.split, img_size=self.img_size
        )

        df = self._read_data()
        _str_to_array = [
            np.fromstring(val, dtype=int, sep=" ") for val in df["pixels"].values
        ]
        self._samples = np.array(_str_to_array)

        if split == "test":
            self._labels = np.empty(shape=len(self._samples))
        else:
            self._labels = df["emotion"].values

    def _read_data(self):
        base_folder = pathlib.Path(self.root) / "data"

        # train y val comparten train.csv, test usa test.csv
        _split = "train" if self.split in ("train", "val") else "test"
        file_name = f"{_split}.csv"
        data_file = base_folder / file_name

        if not os.path.isfile(data_file.as_posix()):
            raise RuntimeError(
                f"{file_name} no se encontró en {base_folder}. "
                "Descarga FER2013 desde Kaggle."
            )

        df = pd.read_csv(data_file)
        if self.split != "test":
            split_index = json.load(open(base_folder / "split.json", "r"))
            split_samples = split_index[self.split]
            df = df.iloc[split_samples]
        return df

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        _vector_img = self._samples[idx]

        # Imagen original 48x48 en escala de grises
        sample_image = _vector_img.reshape(self.img_size, self.img_size).astype(
            "uint8"
        )

        if self.transform is not None:
            image = self.transform(sample_image)  # tensor float32 normalizado
        else:
            image = torch.from_numpy(sample_image)  # uint8

        target = self._labels[idx]
        if self.split != "test":
            emotion = EMOTIONS_MAP[int(target)]
        else:
            emotion = ""

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            "transformed": image,       # tensor que entra a la red
            "label": target,
            "original": sample_image,   # numpy 48x48
            "emotion": emotion,
        }
