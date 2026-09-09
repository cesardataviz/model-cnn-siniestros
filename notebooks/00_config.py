# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 00 · Configuración compartida
# MAGIC
# MAGIC Este notebook no se ejecuta solo: se importa con `%run ./00_config` desde el resto de
# MAGIC notebooks (01-07). Centraliza catálogo, esquema, volumen, tablas y los hiperparámetros
# MAGIC que cambian entre Free Edition (CPU) y un workspace con GPU.
# MAGIC
# MAGIC No pongas secretos aquí (host/token de Databricks van en GitHub Secrets / Databricks Secrets,
# MAGIC nunca en el notebook).

# COMMAND ----------

import os

# --- Unity Catalog ---
CATALOG = "workspace"                    # catálogo default de Free Edition
SCHEMA = "car_damage_mlops"
VOLUME_NAME = "images"                   # /Volumes/workspace/car_damage_mlops/images

# --- Tablas Delta ---
TABLE_IMAGES_METADATA = f"{CATALOG}.{SCHEMA}.images_metadata"   # path, label, split, hash, dims
TABLE_FEATURES = f"{CATALOG}.{SCHEMA}.image_features"           # Feature Store: embeddings + stats

# --- Volume (el "bucket" de Databricks) ---
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

# --- MLflow ---
EXPERIMENT_PATH = "/Shared/car_damage_mlops"
MODEL_NAME_CNN = f"{CATALOG}.{SCHEMA}.car_damage_cnn_scratch"
MODEL_NAME_RESNET = f"{CATALOG}.{SCHEMA}.car_damage_resnet18"

# --- Serving ---
ENDPOINT_NAME_CNN = "car-damage-cnn-scratch"
ENDPOINT_NAME_RESNET = "car-damage-resnet18"

# --- Clases (orden ordinal explícito) ---
CLASS_NAMES = ["minor", "moderate", "severe"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# --- Hiperparámetros de entrenamiento ---
# Free Edition = serverless CPU. IMG_SIZE/EPOCHS bajos por defecto para que una corrida
# completa tome minutos, no horas. Si corres en un workspace con GPU, sube estos valores
# vía variables de entorno antes de %run este notebook.
IMG_SIZE = int(os.environ.get("CAR_DAMAGE_IMG_SIZE", 160))
BATCH_SIZE = int(os.environ.get("CAR_DAMAGE_BATCH_SIZE", 32))
EPOCHS_CNN = int(os.environ.get("CAR_DAMAGE_EPOCHS_CNN", 15))
EPOCHS_RESNET_PER_STAGE = int(os.environ.get("CAR_DAMAGE_EPOCHS_RESNET_STAGE", 4))
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SIMPLE_MEAN = [0.5, 0.5, 0.5]
SIMPLE_STD = [0.5, 0.5, 0.5]

print(f"Catálogo: {CATALOG} | Esquema: {SCHEMA} | Volumen: {VOLUME_PATH}")
print(f"IMG_SIZE={IMG_SIZE} EPOCHS_CNN={EPOCHS_CNN} EPOCHS_RESNET_PER_STAGE={EPOCHS_RESNET_PER_STAGE}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME_NAME}")
print("Schema y Volume verificados/creados.")

# COMMAND ----------

print("config_notebook.py cargado")