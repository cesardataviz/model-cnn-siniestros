# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 08 · Añadir Grad-CAM al servicio (sin reentrenar)
# MAGIC
# MAGIC **Objetivo:** re-loguear ambos modelos Champion con el wrapper pyfunc actualizado
# MAGIC (que ahora también calcula y devuelve el overlay de Grad-CAM en cada predicción),
# MAGIC reutilizando los pesos ya entrenados -- sin volver a correr el entrenamiento
# MAGIC completo de los notebooks 03/04.
# MAGIC
# MAGIC Ejecuta después de que 03-07 ya hayan corrido exitosamente al menos una vez.
# MAGIC **Vuelve a correr el notebook 07 después de este** para que los endpoints sirvan
# MAGIC la nueva versión.

# COMMAND ----------

# MAGIC %pip install torch torchvision mlflow pillow pandas numpy -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import base64
import os
import sys

sys.path.append(os.path.abspath("../src"))

import mlflow
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature

from car_damage.gradcam import GradCAM
from car_damage.models import CarDamageCNN, build_resnet_model_staged
from car_damage.pyfunc_wrapper import CarDamagePyfunc
from car_damage.transforms import make_transforms

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
device = torch.device("cpu")

sample_row = spark.table(TABLE_IMAGES_METADATA).where("split = 'test'").limit(1).collect()[0]
with open(sample_row.path, "rb") as f:
    sample_b64 = base64.b64encode(f.read()).decode("utf-8")
input_example = pd.DataFrame({"image_base64": [sample_b64]})

# COMMAND ----------

def find_state_dict_artifact(run_id: str, path: str = "model") -> str:
    """Busca recursivamente el artifact de pesos del modelo (.pt o state_dict) dentro de los artifacts del run,
    sin asumir una estructura de carpetas fija (puede variar entre versiones de MLflow)."""
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


def relog_with_gradcam(model_type: str, model_name: str) -> int:
    current = client.get_model_version_by_alias(model_name, "Champion")
    source_run_id = current.run_id
    print(f"{model_name}: Champion actual = v{current.version} (run {source_run_id})")

    artifact_path = find_state_dict_artifact(source_run_id)
    if artifact_path is None:
        raise FileNotFoundError(f"No se encontró el artifact 'state_dict' en el run {source_run_id}")
    local_state_dict = mlflow.artifacts.download_artifacts(run_id=source_run_id, artifact_path=artifact_path)

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

    # Para GradCAM, el target_layer necesita gradientes activos incluso si el modelo
    # fue congelado durante entrenamiento
    for param in target_layer.parameters():
        param.requires_grad = True

    eval_tf = make_transforms(mean, std, IMG_SIZE, train=False)
    gradcam = GradCAM(model, target_layer=target_layer)

    # Wrapper real que se sirve (carga sus pesos vía load_context en el endpoint)
    pyfunc_model = CarDamagePyfunc(model_type=model_type, img_size=IMG_SIZE)

    # Wrapper "de prueba" solo para inferir la signature, reusando el modelo ya en memoria
    signature_probe = CarDamagePyfunc(model_type=model_type, img_size=IMG_SIZE)
    signature_probe.model, signature_probe.transform, signature_probe.device = model, eval_tf, device
    signature_probe.gradcam = gradcam
    output_example = pd.DataFrame([signature_probe._predict_one(sample_b64)])
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run(run_name=f"{model_type}_upgrade_gradcam") as run:
        mlflow.set_tag("relog_source_run_id", source_run_id)
        mlflow.set_tag("upgrade", "add_gradcam_overlay")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc_model,
            artifacts={"state_dict": local_state_dict},
            pip_requirements=["torch", "torchvision", "pillow", "pandas", "numpy"],
            code_paths=[os.path.abspath("../src/car_damage")],
            signature=signature,
            input_example=input_example,
        )
        new_run_id = run.info.run_id

    mv = mlflow.register_model(model_uri=f"runs:/{new_run_id}/model", name=model_name)
    client.set_registered_model_alias(name=model_name, alias="Champion", version=mv.version)
    print(f"{model_name}: nueva versión v{mv.version} (con Grad-CAM) -> alias Champion")
    return mv.version


cnn_new_version = relog_with_gradcam("scratch", MODEL_NAME_CNN)
resnet_new_version = relog_with_gradcam("resnet", MODEL_NAME_RESNET)

# COMMAND ----------

# MAGIC %md ## Smoke test: confirmar que la respuesta ahora incluye el overlay de Grad-CAM

# COMMAND ----------

test_df = pd.DataFrame({"image_base64": [sample_b64]})

for name in (MODEL_NAME_CNN, MODEL_NAME_RESNET):
    model = mlflow.pyfunc.load_model(f"models:/{name}@Champion")
    
    # Fix: habilitar gradientes en el modelo después de cargar, ya que load_context()
    # en pyfunc_wrapper.py puede tener capas congeladas del entrenamiento
    py_model = model._model_impl.python_model
    if hasattr(py_model, 'model'):
        for param in py_model.model.parameters():
            param.requires_grad = True
    
    pred = model.predict(test_df)
    row = pred.iloc[0]
    overlay_len = len(row["gradcam_overlay_base64"])
    print(f"\n{name} @Champion")
    print(f"  Predicho: {row['predicted_class']} ({row['confidence']:.1%})")
    print(f"  gradcam_overlay_base64: {overlay_len:,} caracteres (¿no vacío? {overlay_len > 0})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre
# MAGIC Ambos modelos quedan re-registrados (nueva versión, mismo nombre, alias `Champion`
# MAGIC reasignado) sin haber vuelto a entrenar. **Siguiente paso obligatorio: re-ejecutar
# MAGIC el notebook 07** para que los 2 endpoints de Serving sirvan esta versión nueva.