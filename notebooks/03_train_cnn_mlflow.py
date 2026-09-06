# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Entrenamiento CNN desde cero + MLflow
# MAGIC
# MAGIC **Objetivo:** entrenar `CarDamageCNN` con Data Augmentation y class-balanced sampling,
# MAGIC y registrar en MLflow: lineage del dataset (versión Delta), hiperparámetros, curvas por
# MAGIC época, métricas de test (incluyendo Cohen's Kappa cuadrático), matriz de confusión y el
# MAGIC modelo servible (pyfunc que recibe imágenes en base64).
# MAGIC
# MAGIC Ejecuta después de los notebooks 01 y 02.

# COMMAND ----------

# MAGIC %pip install torch torchvision scikit-learn matplotlib seaborn mlflow -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import base64
import os
import sys

sys.path.append(os.path.abspath("../src"))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader

from car_damage.data import ListImageDataset, make_weighted_sampler
from car_damage.gradcam import GradCAM
from car_damage.models import CarDamageCNN
from car_damage.pyfunc_wrapper import CarDamagePyfunc
from car_damage.training import train_model
from car_damage.transforms import denormalize, make_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"Device: {device}")

# COMMAND ----------

# MAGIC %md ## Cargar splits desde Delta (lineage) y armar dataloaders

# COMMAND ----------

history_delta = spark.sql(f"DESCRIBE HISTORY {TABLE_IMAGES_METADATA} LIMIT 1").collect()
delta_version = history_delta[0]["version"]

meta_pd = spark.table(TABLE_IMAGES_METADATA).toPandas()

def samples_for_split(split_name: str) -> list[tuple[str, int]]:
    sub = meta_pd[meta_pd["split"] == split_name]
    return list(zip(sub["path"], sub["label_idx"]))

train_samples = samples_for_split("train")
val_samples = samples_for_split("val")
test_samples = samples_for_split("test")
print(f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} "
      f"(Delta version={delta_version})")

train_tf = make_transforms(SIMPLE_MEAN, SIMPLE_STD, IMG_SIZE, train=True)
eval_tf = make_transforms(SIMPLE_MEAN, SIMPLE_STD, IMG_SIZE, train=False)

train_ds = ListImageDataset(train_samples, train_tf)
val_ds = ListImageDataset(val_samples, eval_tf)
test_ds = ListImageDataset(test_samples, eval_tf)

# Balanceo por sampler (no por weight= en la loss -- aplicar ambos sobre-corregiría)
sampler = make_weighted_sampler(train_samples, num_classes=len(CLASS_NAMES))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# COMMAND ----------

# MAGIC %md ## Entrenar y registrar en MLflow

# COMMAND ----------

with mlflow.start_run(run_name="cnn_scratch") as run:
    mlflow.log_param("architecture", "CarDamageCNN_scratch")
    mlflow.log_param("img_size", IMG_SIZE)
    mlflow.log_param("epochs_max", EPOCHS_CNN)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("delta_table", TABLE_IMAGES_METADATA)
    mlflow.log_param("delta_version", delta_version)

    dataset = mlflow.data.from_pandas(
        meta_pd, source=TABLE_IMAGES_METADATA, name=f"images_metadata_v{delta_version}"
    )
    mlflow.log_input(dataset, context="training")

    model = CarDamageCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history, ckpt_path = train_model(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        epochs=EPOCHS_CNN, patience=5, tag="cnn_scratch", device=device,
    )

    for epoch in range(len(history["train_loss"])):
        mlflow.log_metrics({
            "train_loss": history["train_loss"][epoch],
            "val_loss": history["val_loss"][epoch],
            "train_acc": history["train_acc"][epoch],
            "val_acc": history["val_acc"][epoch],
        }, step=epoch)

    # --- Evaluación en test ---
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    mlflow.log_metric("test_accuracy", report["accuracy"])
    mlflow.log_metric("test_macro_f1", report["macro avg"]["f1-score"])
    mlflow.log_metric("test_quadratic_kappa", kappa)
    print(f"Test accuracy={report['accuracy']:.4f} macro_f1={report['macro avg']['f1-score']:.4f} "
          f"kappa_cuadratico={kappa:.4f}")

    # --- Matriz de confusión como artefacto ---
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title("CNN desde cero")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # --- Grad-CAM: 2 correctos + 2 mal clasificados, como evidencia de auditoría ---
    gradcam = GradCAM(model, target_layer=model.features[-1][0])
    correct_idx = np.where(y_true == y_pred)[0][:2]
    wrong_idx = np.where(y_true != y_pred)[0][:2]

    for tag, indices in [("correct", correct_idx), ("misclassified", wrong_idx)]:
        for i, idx in enumerate(indices):
            img_tensor, true_label = test_ds[idx]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            cam, pred_class, probs = gradcam.generate(input_tensor)
            img = denormalize(img_tensor, SIMPLE_MEAN, SIMPLE_STD)

            fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
            axes[0].imshow(img); axes[0].set_title(f"Real: {CLASS_NAMES[true_label]}"); axes[0].axis("off")
            axes[1].imshow(img); axes[1].imshow(cam, cmap="jet", alpha=0.5)
            axes[1].set_title(f"Pred: {CLASS_NAMES[pred_class]} ({probs[pred_class]:.1%})")
            axes[1].axis("off")
            mlflow.log_figure(fig, f"gradcam/{tag}_{i}.png")
            plt.close(fig)

    # --- Modelo servible (pyfunc, recibe base64) ---
    # Unity Catalog exige signature para registrar un modelo. La construimos aquí mismo,
    # reusando el `model` ya entrenado en memoria (sin recargar desde disco) para que no
    # haga falta duplicar el preprocesamiento ni volver a instalar nada.
    torch.save(model.state_dict(), ckpt_path)
    pyfunc_model = CarDamagePyfunc(model_type="scratch", img_size=IMG_SIZE)

    sample_path, _ = test_samples[0]
    with open(sample_path, "rb") as f:
        sample_b64 = base64.b64encode(f.read()).decode("utf-8")
    input_example = pd.DataFrame({"image_base64": [sample_b64]})

    signature_probe = CarDamagePyfunc(model_type="scratch", img_size=IMG_SIZE)
    signature_probe.model, signature_probe.transform, signature_probe.device = model, eval_tf, device
    output_example = pd.DataFrame([signature_probe._predict_one(sample_b64)])
    signature = infer_signature(input_example, output_example)

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=pyfunc_model,
        artifacts={"state_dict": ckpt_path},
        pip_requirements=["torch", "torchvision", "pillow", "pandas", "numpy"],
        code_paths=[os.path.abspath("../src/car_damage")],
        signature=signature,
        input_example=input_example,
    )

    run_id_cnn = run.info.run_id
    print(f"Run registrado: {run_id_cnn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre
# MAGIC El run queda completo: dataset con versión Delta, curvas por época, métricas de negocio
# MAGIC (Kappa cuadrático como criterio principal, no accuracy), evidencia visual de Grad-CAM y el
# MAGIC modelo listo para registrar en Unity Catalog (notebook 06).