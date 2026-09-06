# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Feature Engineering
# MAGIC
# MAGIC **Objetivo:** derivar features reutilizables por imagen y publicarlas en una tabla de
# MAGIC Feature Store con `image_id` como clave primaria: un embedding de 512 dims de un
# MAGIC ResNet18 congelado (representación semántica genérica) + estadísticas de imagen simples
# MAGIC (brillo, contraste, nitidez, aspect ratio). Esto habilita, más adelante, detección de
# MAGIC drift entre el dataset de entrenamiento y las imágenes reales que lleguen en producción.
# MAGIC
# MAGIC Ejecuta después del notebook 01.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering torch torchvision -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

device = torch.device("cpu")

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()   # nos quedamos con el embedding de 512 dims, sin clasificar
backbone.eval().to(device)

embed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# COMMAND ----------

def laplacian_sharpness(gray_img: np.ndarray) -> float:
    """Varianza del Laplaciano: proxy estándar de nitidez/foco de una imagen."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    from scipy.signal import convolve2d
    lap = convolve2d(gray_img, kernel, mode="valid")
    return float(lap.var())


def compute_row_features(path: str, image_id: int) -> dict:
    img = Image.open(path).convert("RGB")
    gray = np.array(img.convert("L"), dtype=np.float32)

    tensor = embed_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = backbone(tensor).squeeze(0).cpu().numpy()

    width, height = img.size
    return {
        "image_id": image_id,
        "embedding": embedding.tolist(),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": laplacian_sharpness(gray),
        "aspect_ratio": float(width / height),
    }

# COMMAND ----------

rows_pd = spark.table(TABLE_IMAGES_METADATA).select("image_id", "path").toPandas()
feature_rows = [compute_row_features(r.path, r.image_id) for r in rows_pd.itertuples()]

import pandas as pd
features_df = spark.createDataFrame(pd.DataFrame(feature_rows))
print(f"Features calculados para {features_df.count()} imágenes")

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

if not spark.catalog.tableExists(TABLE_FEATURES):
    fe.create_table(
        name=TABLE_FEATURES,
        primary_keys=["image_id"],
        df=features_df,
        description="Embedding ResNet18 (512d, backbone congelado) + estadísticas de imagen "
                     "por image_id, usado para features de auditoría/drift del clasificador "
                     "de severidad de daños vehiculares.",
    )
else:
    fe.write_table(name=TABLE_FEATURES, df=features_df, mode="overwrite")

display(spark.table(TABLE_FEATURES).limit(5))

# COMMAND ----------

print("verificacion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre
# MAGIC La tabla de Feature Store queda disponible para joins por `image_id` en el entrenamiento
# MAGIC (notebooks 03/04) y para comparar distribuciones (brillo, nitidez, embeddings) entre el
# MAGIC dataset de entrenamiento y las imágenes reales que lleguen por la webapp -- la base de
# MAGIC cualquier monitoreo de drift futuro.