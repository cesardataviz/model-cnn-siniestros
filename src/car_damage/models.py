"""Arquitecturas del clasificador de severidad de daños vehiculares.

Portado desde Proyecto_Final_DL_Car_Damage_Colab.ipynb, con un fix: la celda
original tenía `edef set_trainable(...)` (typo, no compilaba).
"""
import torch
import torch.nn as nn
from torchvision import models


class CarDamageCNN(nn.Module):
    """CNN entrenada desde cero. Usa Global Average Pooling (no Flatten+Dense)
    antes de la capa final: son matemáticamente equivalentes a los pesos que
    Grad-CAM calcula vía gradientes (Selvaraju et al. 2020, Sec. 3.1)."""

    def __init__(self, num_classes: int = 3):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def set_trainable(model: nn.Module, layer_prefixes: list[str]) -> None:
    """Congela todo excepto los parámetros cuyo nombre empiece con algún prefijo."""
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(p) for p in layer_prefixes)


def build_resnet_model_staged(device: torch.device, num_classes: int = 3) -> nn.Module:
    """ResNet18 preentrenado en ImageNet, todo congelado salvo `fc` (etapa inicial).
    El fine-tuning gradual de layer3/layer4 se controla luego con `set_trainable`."""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    for p in m.fc.parameters():
        p.requires_grad = True
    return m.to(device)
