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

        # ------------------------------------------------------------------
        # TODO: Calcular dimension de salida
        # En este caso, la "dimension de salida" son las features que salen
        # del backbone antes de la capa fully-connected final.
        # En ResNet18 esa dimensión es backbone.fc.in_features (típicamente 512).
        # ------------------------------------------------------------------
        # Construimos el backbone preentrenado
        self.backbone = build_backbone(model="resnet18", weights="imagenet", freeze=True)

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

        # ------------------------------------------------------------------
        # TODO: Define las capas de tu red
        # Usamos el backbone ResNet18 y reemplazamos SOLO la capa final
        # para que tenga n_classes salidas (7 emociones).
        # ------------------------------------------------------------------
        self.backbone.fc = nn.Linear(out_dim, n_classes)

        # Asegurarnos de que la nueva capa fc sí se entrene
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        self.to(self.device)

    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = x.to(self.device)          # (B, 1, 48, 48)
        logits = self.backbone(x)      # (B, n_classes)
        proba = F.softmax(logits, dim=1)
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        """
        Guarda el modelo en el path especificado
        args:
        - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
        - path (str): path relativo donde se guardará el modelo
        """
        models_path = file_path / "models" / model_name
        if not models_path.parent.exists():
            models_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        """
        Carga el modelo en el path especificado
        args:
        - path (str): path relativo donde se guardó el modelo
        """
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / "models" / model_name
        state_dict = torch.load(models_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)
        self.eval()