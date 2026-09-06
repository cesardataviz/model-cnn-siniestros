# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Model Serving: dos endpoints REST
# MAGIC
# MAGIC **Objetivo:** desplegar los dos `Champion` (CNN desde cero y ResNet18) en dos endpoints
# MAGIC de Databricks Model Serving, con scale-to-zero para no consumir cómputo cuando la webapp
# MAGIC no los está usando. Al final del notebook queda un smoke test real vía REST -- lo mismo
# MAGIC que va a llamar la webapp en Vercel.
# MAGIC
# MAGIC Ejecuta después del notebook 06.

# COMMAND ----------

# MAGIC %pip install databricks-sdk -U -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()


def get_champion_version(model_name: str) -> str:
    from mlflow import MlflowClient
    client = MlflowClient(registry_uri="databricks-uc")
    mv = client.get_model_version_by_alias(model_name, "Champion")
    return mv.version


def upsert_endpoint(endpoint_name: str, model_name: str, entity_name: str):
    version = get_champion_version(model_name)
    print(f"{endpoint_name} <- {model_name} v{version}")

    served_entities = [
        ServedEntityInput(
            entity_name=model_name,
            entity_version=version,
            name=entity_name,
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ]

    existing = [e for e in w.serving_endpoints.list() if e.name == endpoint_name]
    if existing:
        w.serving_endpoints.update_config_and_wait(
            name=endpoint_name, served_entities=served_entities
        )
    else:
        w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(served_entities=served_entities),
        )

    status = w.serving_endpoints.get(endpoint_name)
    print(f"  Estado: {status.state.ready}")
    return status


status_cnn = upsert_endpoint(ENDPOINT_NAME_CNN, MODEL_NAME_CNN, "cnn-scratch-champion")
status_resnet = upsert_endpoint(ENDPOINT_NAME_RESNET, MODEL_NAME_RESNET, "resnet18-champion")

# COMMAND ----------

# MAGIC %md ## Smoke test REST (payload base64, igual al que envía la webapp)

# COMMAND ----------

import base64

sample_row = spark.table(TABLE_IMAGES_METADATA).where("split = 'test'").limit(1).collect()[0]
with open(sample_row.path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {"dataframe_records": [{"image_base64": image_b64}]}

for endpoint_name in (ENDPOINT_NAME_CNN, ENDPOINT_NAME_RESNET):
    response = w.serving_endpoints.query(name=endpoint_name, dataframe_records=payload["dataframe_records"])
    pred = response.predictions[0]
    print(f"\n{endpoint_name}")
    print(f"  Real: {sample_row.label}")
    print(f"  Respuesta: {pred}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre
# MAGIC
# MAGIC ```text
# MAGIC Kaggle -> Volume + Delta -> Features -> MLflow (CNN y ResNet18) -> Unity Catalog (2 Champion)
# MAGIC   -> 2 endpoints REST -> webapp en Vercel
# MAGIC ```
# MAGIC
# MAGIC Ambos endpoints reciben `{"dataframe_records": [{"image_base64": "..."}]}` y devuelven
# MAGIC `{"predicted_class", "confidence", "probabilities"}`. La webapp guarda las URLs de estos
# MAGIC dos endpoints y el token de Databricks como variables de entorno del servidor (nunca en
# MAGIC el cliente).