"""
This file is used to load the FER2013 dataset.
It consists of 48x48 pixel grayscale images of faces
with 7 emotions - angry, disgust, fear, happy, sad, surprise, and neutral.
"""

import pathlib
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import cv2
from ml_clases.proyectos.P02_facial_expressions.utils import to_numpy, add_img_text

file_path = pathlib.Path(__file__).parent.absolute()


def get_loader(split, batch_size, shuffle=True, num_workers=0):
    """
    Get train and validation loaders
    args:
        - batch_size (int): batch size
        - split (str): split to load (train, test or val)
    """
    _training = split == "train"
    dataset = datasets.MNIST(
        root="./mnist/train",
        train=_training,
        # download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, dataloader


def main():
    # Visualizar de una en una imagen
    split = "train"
    dataset, dataloader = get_loader(split=split, batch_size=1, shuffle=False)
    print(f"Loading {split} set with {len(dataloader)} samples")
    for datapoint in dataloader:
        img, label = datapoint
        label = to_numpy(label, roll_dims=False).item()

        # Transformar a numpy
        original = to_numpy(img)  # 0 - 1.0 float

        # Aumentar el tamaño de la imagen para visualizarla mejor
        viz_size = (200, 200)
        original = cv2.resize(original, viz_size)

        # Concatenar las imagenes, tienen que ser del mismo tipo
        np_img = add_img_text(original, str(label))

        cv2.imshow("img", np_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
