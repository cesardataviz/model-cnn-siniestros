# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingesta a Unity Catalog Volume + Delta
# MAGIC
# MAGIC **Objetivo:** bajar el dataset de Kaggle, copiar las imágenes al Volume de Unity Catalog
# MAGIC (el "bucket" de Databricks) y registrar en Delta los metadatos con split e integridad
# MAGIC (hash), **nunca los píxeles** -- eso es lo correcto para imágenes reales, a diferencia
# MAGIC de la demo de Fashion-MNIST que sí guarda bytes en Delta por ser un dataset de juguete.
# MAGIC
# MAGIC Ejecuta en compute Serverless de Free Edition.

# COMMAND ----------

# MAGIC %pip install kagglehub -q
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import hashlib
import os
import shutil

import kagglehub
from PIL import Image
from pyspark.sql import Row

path = kagglehub.dataset_download("prajwalbhamere/car-damage-severity-dataset")
print("Dataset descargado en:", path)

TRAIN_DIR = os.path.join(path, "data3a", "training")
VAL_DIR = os.path.join(path, "data3a", "validation")

# COMMAND ----------

import sys
sys.path.append(os.path.abspath("../src"))
from car_damage.data import collect_samples

kaggle_train_samples = collect_samples(TRAIN_DIR, CLASS_TO_IDX)
kaggle_val_samples = collect_samples(VAL_DIR, CLASS_TO_IDX)

print(f"training/: {len(kaggle_train_samples)} imágenes")
print(f"validation/: {len(kaggle_val_samples)} imágenes")

# COMMAND ----------

from sklearn.model_selection import train_test_split

kaggle_val_labels = [lbl for _, lbl in kaggle_val_samples]
val_samples, test_samples = train_test_split(
    kaggle_val_samples, test_size=0.5, stratify=kaggle_val_labels, random_state=SEED
)
train_samples = kaggle_train_samples

splits = {"train": train_samples, "val": val_samples, "test": test_samples}
for name, s in splits.items():
    print(f"{name}: {len(s)} imágenes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Copiar imágenes al Volume y armar los metadatos
# MAGIC
# MAGIC Layout en el Volume: `{VOLUME_PATH}/{split}/{clase}/{image_id}.jpg`

# COMMAND ----------

def sha256_of_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
image_id = 0
for split_name, samples in splits.items():
    for src_path, label_idx in samples:
        class_name = CLASS_NAMES[label_idx]
        dest_dir = os.path.join(VOLUME_PATH, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        ext = os.path.splitext(src_path)[1].lower()
        dest_filename = f"{image_id:06d}{ext}"
        dest_path = os.path.join(dest_dir, dest_filename)
        shutil.copyfile(src_path, dest_path)

        with Image.open(src_path) as im:
            width, height = im.size

        rows.append(Row(
            image_id=image_id,
            path=dest_path,
            split=split_name,
            label=class_name,
            label_idx=label_idx,
            width=width,
            height=height,
            sha256=sha256_of_file(dest_path),
            bytes_size=os.path.getsize(dest_path),
        ))
        image_id += 1

print(f"Copiadas {image_id} imágenes a {VOLUME_PATH}")

# COMMAND ----------

df_metadata = spark.createDataFrame(rows)
(
    df_metadata.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_IMAGES_METADATA)
)

display(spark.sql(f"""
    SELECT split, label, count(*) as n
    FROM {TABLE_IMAGES_METADATA}
    GROUP BY split, label
    ORDER BY split, label
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre
# MAGIC Las imágenes viven en el Volume (versionable, auditable, con control de acceso de Unity
# MAGIC Catalog). La tabla Delta versiona el *índice*: qué imagen, en qué split, con qué hash --
# MAGIC suficiente para reproducir exactamente el dataset usado en cualquier entrenamiento futuro.
