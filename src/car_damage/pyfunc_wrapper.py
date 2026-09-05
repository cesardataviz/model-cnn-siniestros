"""Wrapper MLflow pyfunc para servir el clasificador vía REST.

Por qué existe: el endpoint de Model Serving recibe JSON, no tensores PyTorch.
Si sirviéramos el tensor crudo, cualquier cliente (incluida la webapp) tendría
que replicar exactamente nuestro preprocesamiento -- imposible de mantener.
Este wrapper recibe una imagen en base64, hace decode + resize + normalize
internamente, y devuelve {clase, probabilidades, confianza} ya listo para la UI.
"""
import base64
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .models import CarDamageCNN, build_resnet_model_staged
from .transforms import make_transforms

CLASS_NAMES = ["minor", "moderate", "severe"]


class CarDamagePyfunc:
    """mlflow.pyfunc.PythonModel. `model_type` es "scratch" o "resnet" -- decide
    qué arquitectura reconstruir y con qué estadísticas de normalización."""

    def __init__(self, model_type: str, img_size: int = 224):
        if model_type not in ("scratch", "resnet"):
            raise ValueError('model_type debe ser "scratch" o "resnet"')
        self.model_type = model_type
        self.img_size = img_size

    def load_context(self, context):
        device = torch.device("cpu")
        state_dict = torch.load(context.artifacts["state_dict"], map_location=device)

        if self.model_type == "scratch":
            self.model = CarDamageCNN(num_classes=len(CLASS_NAMES))
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            self.model = build_resnet_model_staged(device, num_classes=len(CLASS_NAMES))
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()
        self.transform = make_transforms(mean, std, self.img_size, train=False)
        self.device = device

    def _predict_one(self, image_b64: str) -> dict:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        }

    def predict(self, context, model_input: pd.DataFrame, params=None):
        images_b64 = model_input["image_base64"].tolist()
        return pd.DataFrame([self._predict_one(b64) for b64 in images_b64])
