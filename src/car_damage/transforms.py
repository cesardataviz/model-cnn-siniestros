"""Pipelines de normalización y augmentation.

Dos normalizaciones distintas por diseño: el CNN desde cero usa (0.5, 0.5, 0.5)
porque no hay pesos preentrenados de por medio; ResNet18 usa las estadísticas de
ImageNet porque el backbone preentrenado espera esa distribución de entrada.
"""
import torch
from torchvision import transforms


def make_transforms(mean: list[float], std: list[float], img_size: int, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((int(img_size * 256 / 224), int(img_size * 256 / 224))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def denormalize(tensor: torch.Tensor, mean: list[float], std: list[float]):
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = tensor.cpu() * std_t + mean_t
    return img.permute(1, 2, 0).clamp(0, 1).numpy()
