from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from ml25.P02_facial_expressions.dataset import get_loader
from ml25.P02_facial_expressions.network import Network
from ml25.P02_facial_expressions.plot_losses import PlotLosses  # <-- agregado

# Logging
import wandb
from datetime import datetime, timezone


def init_wandb(cfg):
    # Initialize wandb
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    run = wandb.init(
        project="facial_expressions_cnn",
        config=cfg,
        name=f"facial_expressions_cnn_{timestamp}_utc",
    )
    return run


def validation_step(val_loader, net, cost_function):
    """
    Realiza un epoch completo en el conjunto de validación
    args:
    - val_loader (torch.DataLoader): dataloader para los datos de validación
    - net: instancia de red neuronal de clase Network
    - cost_function (torch.nn): Función de costo a utilizar

    returns:
    - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    """
    val_loss = 0.0
    device = net.device
    net.eval()
    with torch.inference_mode():
        for i, batch in enumerate(val_loader, 0):
            batch_imgs = batch["transformed"].to(device)
            batch_labels = batch["label"].to(device)

            # forward + loss
            logits, proba = net(batch_imgs)
            loss = cost_function(logits, batch_labels)
            val_loss += loss.item()

    # Regresa el costo promedio por minibatch
    return val_loss / len(val_loader)


def train():
    # Hyperparametros (mejorados para CPU)
    cfg = {
        "training": {
            "learning_rate": 1e-3,
            "n_epochs": 40,   # antes 100
            "batch_size": 128, # antes 256
        },
    }
    run = init_wandb(cfg)

    train_cfg = cfg.get("training")
    learning_rate = train_cfg.get("learning_rate")
    n_epochs = train_cfg.get("n_epochs")
    batch_size = train_cfg.get("batch_size")

    # Train, validation loaders
    train_dataset, train_loader = get_loader(
        "train", batch_size=batch_size, shuffle=True
    )
    val_dataset, val_loader = get_loader("val", batch_size=batch_size, shuffle=False)
    print(
        f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}"
    )

    # Instanciamos tu red
    modelo = Network(input_dim=48, n_classes=7)
    device = modelo.device

    # Funcion de costo
    criterion = nn.CrossEntropyLoss()

    # Define el optimizador (solo params entrenables si congelaste backbone)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()),
        lr=learning_rate,
    )

    # Para graficar loss de train y val
    plotter = PlotLosses()

    best_epoch_loss = np.inf
    for epoch in range(n_epochs):
        modelo.train()
        train_loss = 0.0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch["transformed"].to(device)
            batch_labels = batch["label"].to(device)

            # Zero grad, forward, backward, step
            optimizer.zero_grad()
            logits, proba = modelo(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            # acumula el costo
            train_loss += loss.item()

        # Costo promedio de entrenamiento
        train_loss = train_loss / len(train_loader)
        # Costo en validación
        val_loss = validation_step(val_loader, modelo, criterion)

        # Actualizar gráfica
        plotter.on_epoch_end(epoch, train_loss, val_loss)

        tqdm.write(
            f"Epoch: {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        )

        # guarda el modelo si el costo de validación es menor al mejor costo de validación
        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss
            modelo.save_model("modelo_1.pt")
            print(f"--> Nuevo mejor modelo guardado con val_loss = {val_loss:.4f}")

        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
            }
        )

    # Al final, deja la gráfica en pantalla y la guarda en figures/
    plotter.on_train_end()


if __name__ == "__main__":
    train()
