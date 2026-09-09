# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 09 · Análisis de errores en las esquinas (minor ↔ severe)
# MAGIC
# MAGIC **Objetivo:** identificar y visualizar, para cada modelo, las imágenes de test donde
# MAGIC ocurrió el error más costoso para el negocio -- confundir `minor` con `severe` o
# MAGIC viceversa -- y ver su Grad-CAM para entender en qué se fijó el modelo.
# MAGIC
# MAGIC No reentrena nada: reconstruye cada arquitectura a partir de los pesos ya guardados
# MAGIC del Champion actual (mismo patrón que el notebook 08), y vuelve a evaluar sobre el
# MAGIC conjunto de test para ubicar los casos exactos.
# MAGIC
# MAGIC Ejecuta después de que 01-07 (y opcionalmente 08) ya hayan corrido.

# COMMAND ----------

print("Test set: {len(test_samples)} imágenes")

# COMMAND ----------

# MAGIC %pip install torch torchvision mlflow scikit-learn matplotlib -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from torch.utils.data import DataLoader

from car_damage.data import ListImageDataset
from car_damage.gradcam import GradCAM
from car_damage.models import CarDamageCNN, build_resnet_model_staged
from car_damage.transforms import denormalize, make_transforms

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
device = torch.device("cpu")

meta_pd = spark.table(TABLE_IMAGES_METADATA).toPandas()
test_sub = meta_pd[meta_pd["split"] == "test"]
test_samples = list(zip(test_sub["path"], test_sub["label_idx"]))
print(f"Test set: {len(test_samples)} imágenes")

# COMMAND ----------

def find_state_dict_artifact(run_id: str, path: str = "model") -> str:
    """Igual que en el notebook 08: busca el archivo de pesos sin asumir una ruta fija."""
    for f in mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path=path):
        if f.is_dir:
            found = find_state_dict_artifact(run_id, f.path)
            if found:
                return found
        else:
            basename = os.path.basename(f.path)
            if basename == "state_dict" or basename.endswith(".pt"):
                return f.path
    return None


def load_champion_model(model_type: str, model_name: str):
    current = client.get_model_version_by_alias(model_name, "Champion")
    artifact_path = find_state_dict_artifact(current.run_id)
    local_state_dict = mlflow.artifacts.download_artifacts(run_id=current.run_id, artifact_path=artifact_path)

    if model_type == "scratch":
        model = CarDamageCNN(num_classes=len(CLASS_NAMES))
        mean, std = SIMPLE_MEAN, SIMPLE_STD
        target_layer = model.features[-1][0]
    else:
        model = build_resnet_model_staged(device, num_classes=len(CLASS_NAMES))
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        target_layer = model.layer4[-1]

    model.load_state_dict(torch.load(local_state_dict, map_location=device))
    model.to(device).eval()
    for param in target_layer.parameters():
        param.requires_grad = True  # necesario para que Grad-CAM capture gradientes

    eval_tf = make_transforms(mean, std, IMG_SIZE, train=False)
    gradcam = GradCAM(model, target_layer=target_layer)
    return model, eval_tf, gradcam, mean, std, current.version

# COMMAND ----------

def evaluate_test_set(model, eval_tf):
    """Corre inferencia (sin gradiente) sobre todo el test set y devuelve y_true/y_pred."""
    test_ds = ListImageDataset(test_samples, eval_tf)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_pred), test_ds


def show_corner_cases(model_name_label, model, eval_tf, gradcam, mean, std, y_true, y_pred, test_ds):
    minor_idx, severe_idx = CLASS_TO_IDX["minor"], CLASS_TO_IDX["severe"]

    cases = [
        ("Real MINOR -> Predicho SEVERE", np.where((y_true == minor_idx) & (y_pred == severe_idx))[0]),
        ("Real SEVERE -> Predicho MINOR", np.where((y_true == severe_idx) & (y_pred == minor_idx))[0]),
    ]

    for title, indices in cases:
        print(f"\n=== {model_name_label} | {title} | {len(indices)} caso(s) ===")
        if len(indices) == 0:
            print("  (ningún caso en esta celda -- buena señal)")
            continue

        n = len(indices)
        fig, axes = plt.subplots(1, n * 2, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes[0], axes[1]]

        for i, idx in enumerate(indices):
            path, true_label = test_samples[idx]
            img_tensor, _ = test_ds[idx]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            cam, pred_class, probs = gradcam.generate(input_tensor)
            img = denormalize(img_tensor, mean, std)

            ax_img, ax_cam = axes[i * 2], axes[i * 2 + 1]
            ax_img.imshow(img)
            ax_img.set_title(f"{os.path.basename(path)}\nReal: {CLASS_NAMES[true_label]}", fontsize=9)
            ax_img.axis("off")

            ax_cam.imshow(img)
            ax_cam.imshow(cam, cmap="jet", alpha=0.5)
            ax_cam.set_title(f"Grad-CAM\nPred: {CLASS_NAMES[pred_class]} ({probs[pred_class]:.1%})", fontsize=9)
            ax_cam.axis("off")

        fig.suptitle(f"{model_name_label} -- {title}")
        plt.tight_layout()
        plt.show()  # se queda visible en la salida de la celda (a diferencia de 03/04, aquí no se cierra)

# COMMAND ----------

# MAGIC %md ## CNN desde cero

# COMMAND ----------

model_s, tf_s, gradcam_s, mean_s, std_s, ver_s = load_champion_model("scratch", MODEL_NAME_CNN)
print(f"{MODEL_NAME_CNN} Champion v{ver_s}")
y_true_s, y_pred_s, test_ds_s = evaluate_test_set(model_s, tf_s)
show_corner_cases("CNN desde cero", model_s, tf_s, gradcam_s, mean_s, std_s, y_true_s, y_pred_s, test_ds_s)

# COMMAND ----------

# MAGIC %md ## ResNet18 (Transfer Learning)

# COMMAND ----------

model_r, tf_r, gradcam_r, mean_r, std_r, ver_r = load_champion_model("resnet", MODEL_NAME_RESNET)
print(f"{MODEL_NAME_RESNET} Champion v{ver_r}")
y_true_r, y_pred_r, test_ds_r = evaluate_test_set(model_r, tf_r)
show_corner_cases("ResNet18 (TL)", model_r, tf_r, gradcam_r, mean_r, std_r, y_true_r, y_pred_r, test_ds_r)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre
# MAGIC Para cada caso mostrado arriba, pregúntate lo mismo que la sección 3.2 del informe:
# MAGIC ¿el Grad-CAM se concentra en una zona de daño real que un humano confundiría igual
# MAGIC (ej. un rayón severo pero pequeño, o una abolladura leve en mala iluminación), o se
# MAGIC concentra en algo irrelevante (fondo, reflejos, color del auto)? La primera situación
# MAGIC es un error "razonable" del modelo; la segunda es evidencia de correlación espuria.