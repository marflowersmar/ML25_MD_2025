import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()


def build_backbone(model="resnet18", weights="imagenet", freeze=True, last_n_layers=2):
    if model == "resnet18":
        # Usar pesos preentrenados en ImageNet
        if weights == "imagenet":
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)

        # Congelar pesos si se indica
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f"Model {model} not supported")


class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Construimos el backbone preentrenado
        self.backbone = build_backbone(
            model="resnet18",
            weights="imagenet",
            freeze=True,
        )

        # Adaptar la primera capa para 1 canal (gris) en lugar de 3 canales (RGB)
        old_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=(old_conv1.bias is not None),
        )
        # Copiar pesos, promediando los 3 canales a 1
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                old_conv1.weight.mean(dim=1, keepdim=True)
            )
            if old_conv1.bias is not None:
                self.backbone.conv1.bias = old_conv1.bias

        # Dimension de las features antes de la capa fc
        out_dim = self.backbone.fc.in_features  # normalmente 512

        # Reemplazamos SOLO la capa final para que tenga n_classes salidas (7 emociones)
        self.backbone.fc = nn.Linear(out_dim, n_classes)

        # Asegurarnos de que la nueva capa fc sí se entrene
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        self.to(self.device)

    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x llega como (B, 1, 48, 48)
        x = x.to(self.device)

        # --- Enfatizar ligeramente la parte inferior de la cara (boca) ---
        # Creamos una máscara vertical que va de 0.7 (arriba) a 1.3 (abajo)
        B, C, H, W = x.shape
        row_weights = torch.linspace(0.7, 1.3, steps=H, device=self.device)  # (H,)
        mask = row_weights.view(1, 1, H, 1)  # (1, 1, H, 1)
        x = x * mask  # la zona inferior (boca) tiene más peso en la activación

        # Pasamos por el backbone ResNet18 adaptado
        logits = self.backbone(x)      # (B, n_classes)
        proba = F.softmax(logits, dim=1)
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        """
        Guarda el modelo en el path especificado
        """
        models_path = file_path / "models" / model_name
        if not models_path.parent.exists():
            models_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        """
        Carga el modelo en el path especificado
        """
        models_path = file_path / "models" / model_name
        state_dict = torch.load(models_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)
        self.eval()
