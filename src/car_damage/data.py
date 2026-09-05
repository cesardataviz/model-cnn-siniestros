"""Utilidades de dataset: mapeo de carpetas Kaggle a clases, Dataset genérico,
y el sampler balanceado que corrige el desbalance de clases.

Nota sobre un bug del notebook original: los comentarios decían "ya usas el
sampler" pero nunca se construía ningún WeightedRandomSampler -- se calculaban
class_weights y no se usaban en ningún lado, por lo que el desbalance no se
corregía. `make_weighted_sampler` es la pieza que faltaba.
"""
import os
import re
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


def normalize_folder_name(name: str) -> str:
    """'01-minor', 'Minor', '02_moderate', 'SEVERE' -> 'minor', 'moderate', 'severe'"""
    name = name.lower().strip()
    name = re.sub(r"^[\d\-_.\s]+", "", name)
    return name.strip()


def collect_samples(root_dir: str, class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
    """Recorre root_dir/<clase>/*.jpg y arma [(ruta_absoluta, indice_de_clase), ...].
    Tolera nombres de carpeta tipo '01-minor', 'Minor', 'MINOR'."""
    samples = []
    if not os.path.isdir(root_dir):
        print(f"[aviso] la carpeta no existe: {root_dir}")
        return samples

    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for folder_name in subfolders:
        normalized = normalize_folder_name(folder_name)
        matched_class = next(
            (cls for cls in class_to_idx if cls in normalized or normalized in cls), None
        )
        if matched_class is None:
            print(f"  [aviso] no pude mapear '{folder_name}' a ninguna clase de "
                  f"{list(class_to_idx.keys())} -- se ignora")
            continue

        idx = class_to_idx[matched_class]
        cls_dir = os.path.join(root_dir, folder_name)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, fname), idx))

    return samples


class ListImageDataset(Dataset):
    """Dataset genérico a partir de una lista [(ruta, etiqueta), ...]."""

    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_weighted_sampler(samples: list[tuple[str, int]], num_classes: int) -> WeightedRandomSampler:
    """Sampler que balancea las clases a nivel de batch (inverso a la frecuencia).
    Se usa en vez de `weight=` en la loss para no aplicar el balanceo dos veces."""
    labels = [lbl for _, lbl in samples]
    counts = Counter(labels)
    class_weight = {c: 1.0 / counts[c] for c in range(num_classes) if counts.get(c, 0) > 0}
    sample_weights = [class_weight[lbl] for lbl in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
