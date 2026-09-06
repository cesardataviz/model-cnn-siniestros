# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Model Registry (Unity Catalog)
# MAGIC
# MAGIC **Objetivo:** registrar **ambos** modelos por separado en Unity Catalog -- cada
# MAGIC arquitectura tiene su propio nombre y su propio alias `Champion` -- porque la webapp va a
# MAGIC exponer los dos endpoints en paralelo (CNN desde cero y ResNet18), no solo "el mejor".
# MAGIC
# MAGIC `Champion` en cada modelo apunta siempre a la última versión con mejor `test_quadratic_kappa`
# MAGIC de esa arquitectura -- así versionamos independientemente cada familia de modelo.
# MAGIC
# MAGIC Ejecuta después del notebook 05.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)

# COMMAND ----------

# DBTITLE 1,Cell 4
def register_best(architecture_param: str, model_name: str) -> tuple[str, int]:
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' and params.architecture = '{architecture_param}'",
        order_by=["metrics.test_quadratic_kappa DESC"],
    )
    if len(runs) == 0:
        raise ValueError(f"No hay runs FINISHED para architecture='{architecture_param}'")

    best_run_id = runs.iloc[0]["run_id"]
    best_kappa = runs.iloc[0]["metrics.test_quadratic_kappa"]
    print(f"Mejor run para {architecture_param}: {best_run_id} (kappa={best_kappa:.4f})")

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_registered_model_alias(name=model_name, alias="Champion", version=mv.version)
    print(f"{model_name} v{mv.version} -> alias Champion")
    return model_name, mv.version


cnn_name, cnn_version = register_best("CarDamageCNN_scratch", MODEL_NAME_CNN)
resnet_name, resnet_version = register_best("resnet18_staged_finetuning", MODEL_NAME_RESNET)

# COMMAND ----------

# MAGIC %md ## Smoke test: cargar cada Champion y predecir sobre una imagen del test set

# COMMAND ----------

import base64

import pandas as pd

sample_row = spark.table(TABLE_IMAGES_METADATA).where("split = 'test'").limit(1).collect()[0]
with open(sample_row.path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

test_df = pd.DataFrame({"image_base64": [image_b64]})

for name in (MODEL_NAME_CNN, MODEL_NAME_RESNET):
    model = mlflow.pyfunc.load_model(f"models:/{name}@Champion")
    pred = model.predict(test_df)
    print(f"\n{name} @Champion")
    print(f"  Real: {sample_row.label}")
    print(f"  Predicho: {pred.iloc[0]['predicted_class']} ({pred.iloc[0]['confidence']:.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre
# MAGIC Ambos modelos quedan gobernados en Unity Catalog, con versión y alias `Champion`
# MAGIC independientes. El notebook 07 despliega cada `Champion` en su propio endpoint REST.