import math
import pathlib
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()


def make_resnet_backbone(
    use_pretrained: bool = True,
    fine_tune_last_block: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Crea un backbone ResNet18 para transfer learning.
    Regresa:
    - backbone sin la capa fc final
    - dimensión de salida del vector de características
    """
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    net = resnet18(weights=weights)

    # Congelar todo al inicio
    for p in net.parameters():
        p.requires_grad = False

    # Si queremos fine-tuning, descongelamos layer4
    if fine_tune_last_block:
        for p in net.layer4.parameters():
            p.requires_grad = True

    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


class Network(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        use_pretrained: bool = True,
        fine_tune_last_block: bool = True,
    ) -> None:
        super().__init__()

        # Solo CPU para tu caso
        self.device = "cpu"

        # Backbone ResNet18
        self.backbone, feat_dim = make_resnet_backbone(
            use_pretrained=use_pretrained,
            fine_tune_last_block=fine_tune_last_block,
        )

        # Classifier más grande: 512 → 512 → 256 → 7
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

        self.to(self.device)

    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        return math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1

    def forward(self, x: torch.Tensor):
        """
        x esperado: (B, 1, 48, 48).
        Replicamos el canal para ResNet (B,3,48,48).
        """
        if x.ndim != 4:
            raise ValueError(f"Se esperaba tensor 4D (B,C,H,W), recibido {x.shape}")

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x = x.to(self.device)
        feats = self.backbone(x)          # (B, feat_dim)
        logits = self.classifier(feats)   # (B, n_classes)
        proba = F.softmax(logits, dim=1)
        return logits, proba

    def predict(self, x: torch.Tensor):
        self.eval()
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        models_path = file_path / "models" / model_name
        models_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), models_path)
        print(f"Modelo guardado en: {models_path}")

    def load_model(self, model_name: str):
        models_path = file_path / "models" / model_name
        if not models_path.exists():
            raise FileNotFoundError(f"No encontré el modelo en: {models_path}")
        state_dict = torch.load(models_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)
        self.eval()
        print(f"Modelo cargado desde: {models_path}")
