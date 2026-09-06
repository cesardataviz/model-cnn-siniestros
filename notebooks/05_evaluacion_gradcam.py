# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Evaluación comparativa y auditoría Grad-CAM
# MAGIC
# MAGIC **Objetivo:** comparar los runs de `cnn_scratch` y `resnet18_transfer_learning`, con
# MAGIC **Cohen's Kappa cuadrático** como criterio principal (no accuracy): las clases
# MAGIC `minor < moderate < severe` son ordinales, y confundir `minor` con `severe` es un error
# MAGIC mucho más costoso para el negocio que confundir `minor` con `moderate`.
# MAGIC
# MAGIC Ejecuta después de los notebooks 03 y 04.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow

experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.test_quadratic_kappa DESC"],
)

cols = ["run_id", "tags.mlflow.runName", "params.architecture",
        "metrics.test_accuracy", "metrics.test_macro_f1", "metrics.test_quadratic_kappa"]
display(runs[cols])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checklist de auditoría de negocio
# MAGIC
# MAGIC Revisar en MLflow (pestaña Artifacts de cada run, carpeta `gradcam/`) para ambos modelos:
# MAGIC
# MAGIC 1. **¿El heatmap se concentra en la zona de daño** (abolladura, rayón, cristal roto)?
# MAGIC 2. **¿El heatmap se concentra en el fondo, el logo o la placa?** → correlación espuria,
# MAGIC    riesgo legal/de negocio inaceptable en un sistema que afecta pagos de seguros.
# MAGIC 3. **En los casos mal clasificados (`gradcam/misclassified_*.png`), ¿el error es razonable?**
# MAGIC    (ej. confundir `moderate` con `severe`, no `minor` con `severe`).
# MAGIC 4. **¿Cuál modelo tiene el Grad-CAM más nítido y localizado?**
# MAGIC
# MAGIC Documentar la respuesta a estas 4 preguntas, con capturas, en el informe -- es el criterio
# MAGIC de "¿es desplegable en su estado actual?" que pide la rúbrica en la sección de Resultados.

# COMMAND ----------

best_cnn = runs[runs["params.architecture"] == "CarDamageCNN_scratch"].iloc[0]
best_resnet = runs[runs["params.architecture"] == "resnet18_staged_finetuning"].iloc[0]

print("Mejor CNN desde cero:", best_cnn["run_id"], "kappa=", best_cnn["metrics.test_quadratic_kappa"])
print("Mejor ResNet18 TL:   ", best_resnet["run_id"], "kappa=", best_resnet["metrics.test_quadratic_kappa"])

dbutils.jobs.taskValues.set(key="best_cnn_run_id", value=best_cnn["run_id"])
dbutils.jobs.taskValues.set(key="best_resnet_run_id", value=best_resnet["run_id"])