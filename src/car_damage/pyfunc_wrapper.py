"""Wrapper MLflow pyfunc para servir el clasificador vía REST.

Por qué existe: el endpoint de Model Serving recibe JSON, no tensores PyTorch.
Si sirviéramos el tensor crudo, cualquier cliente (incluida la webapp) tendría
que replicar exactamente nuestro preprocesamiento -- imposible de mantener.
Este wrapper recibe una imagen en base64, hace decode + resize + normalize
internamente, y devuelve {clase, probabilidades, confianza, overlay Grad-CAM}
ya listo para la UI -- el heatmap se calcula en el servidor por la misma razón:
el cliente no tiene (ni debe tener) acceso a los pesos del modelo.
"""
import base64
import io

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .gradcam import GradCAM
from .models import CarDamageCNN, build_resnet_model_staged
from .transforms import make_transforms

CLASS_NAMES = ["minor", "moderate", "severe"]

# Paradas de color aproximando el colormap "jet" (azul->cian->verde->amarillo->rojo),
# implementado a mano para no depender de matplotlib solo por esto en el endpoint.
_JET_STOPS_POS = [0.0, 0.25, 0.5, 0.75, 1.0]
_JET_STOPS_RGB = np.array([
    [0, 0, 143], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0],
], dtype=np.float32)


def _apply_jet_colormap(cam: np.ndarray) -> np.ndarray:
    """cam: HxW float en [0,1]. Devuelve HxWx3 uint8."""
    flat = cam.flatten()
    channels = [np.interp(flat, _JET_STOPS_POS, _JET_STOPS_RGB[:, c]) for c in range(3)]
    rgb = np.stack(channels, axis=1).reshape(cam.shape[0], cam.shape[1], 3)
    return rgb.astype(np.uint8)


class CarDamagePyfunc(mlflow.pyfunc.PythonModel):
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
            target_layer = self.model.features[-1][0]
        else:
            self.model = build_resnet_model_staged(device, num_classes=len(CLASS_NAMES))
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            target_layer = self.model.layer4[-1]

        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()
        self.transform = make_transforms(mean, std, self.img_size, train=False)
        self.device = device
        
        # Habilitar gradientes en target_layer para que GradCAM capture los gradientes correctamente.
        # build_resnet_model_staged() congela todas las capas excepto 'fc', así que debemos
        # reactivar requires_grad en target_layer antes de crear GradCAM.
        for param in target_layer.parameters():
            param.requires_grad = True
        
        self.gradcam = GradCAM(self.model, target_layer=target_layer)

    def _make_overlay_b64(self, pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> str:
        base = pil_img.resize((self.img_size, self.img_size)).convert("RGB")
        base_arr = np.array(base, dtype=np.float32)
        heat = _apply_jet_colormap(cam).astype(np.float32)
        blended = np.clip((1 - alpha) * base_arr + alpha * heat, 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _predict_one(self, image_b64: str) -> dict:
        raw = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # GradCAM.generate() hace forward + backward y devuelve el heatmap junto
        # con la clase predicha y las probabilidades -- un solo pase basta para
        # clasificar y auditar a la vez.
        cam, pred_idx, probs = self.gradcam.generate(tensor)
        overlay_b64 = self._make_overlay_b64(pil_img, cam)

        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            "gradcam_overlay_base64": overlay_b64,
        }

    def predict(self, context, model_input: pd.DataFrame, params=None):
        images_b64 = model_input["image_base64"].tolist()
        return pd.DataFrame([self._predict_one(b64) for b64 in images_b64])
