# Clasificador de Severidad de Siniestros — MLOps end-to-end

Triage automático de severidad de daños vehiculares (`minor` / `moderate` / `severe`) para
aseguradoras. Cubre los 3 pilares de MLOps: **datos** (Unity Catalog Volumes + Delta + Feature
Store), **modelo** (MLflow + Unity Catalog Registry + Model Serving) y **código** (GitHub con
control de versiones y CI/CD hacia Databricks y Vercel).

## Pipeline

```
Kaggle (car-damage-severity-dataset)
  -> notebooks/01: Unity Catalog Volume (imágenes) + tabla Delta (metadata/split/hash)
  -> notebooks/02: Feature Store (embedding ResNet18 + estadísticas por imagen)
  -> notebooks/03: entrena CNN desde cero -> MLflow (curvas, Kappa cuadrático, Grad-CAM)
  -> notebooks/04: entrena ResNet18 (fine-tuning por etapas) -> MLflow
  -> notebooks/05: comparación de runs + checklist de auditoría Grad-CAM
  -> notebooks/06: registro en Unity Catalog, alias @Champion (uno por arquitectura)
  -> notebooks/07: 2 endpoints de Model Serving (scale-to-zero)
  -> webapp/ (Next.js en Vercel): sube foto, elige modelo, llama al endpoint, muestra severidad
```

Dos modelos, dos endpoints, dos alias `Champion` independientes -- la webapp permite comparar
ambos lado a lado, no solo mostrar "el mejor".

## Estructura del repo

- `notebooks/` — notebooks en formato *source* de Databricks (`.py` con `# COMMAND ----------`).
  Se editan igual en el Workspace de Databricks y en GitHub; el diff es legible en cada PR.
- `src/car_damage/` — código compartido (arquitecturas, transforms, Grad-CAM, dataset, bucle de
  entrenamiento, wrapper pyfunc para servir). Los notebooks importan de aquí para no duplicar código.
- `webapp/` — app Next.js desplegada en Vercel. El route handler `app/api/predict/route.ts` es el
  único lugar que conoce el token de Databricks; el cliente nunca lo ve.
- `.github/workflows/` — CI/CD (ver abajo).
- `docs/` — decisiones de diseño y notas para el informe.

## Origen y decisiones de diseño

Este proyecto parte de `Proyecto_Final_DL_Car_Damage_Colab.ipynb` (CNN desde cero + ResNet18 +
Grad-CAM en PyTorch, entrenado en Colab con GPU). Al portarlo a Databricks se corrigieron 3 bugs
del notebook original:

1. `edef set_trainable(...)` → typo, la celda no compilaba (`src/car_damage/models.py`).
2. La función de prueba puntual invertía los modelos: pedir `"scratch"` cargaba `model_resnet`
   y viceversa.
3. Se calculaban `class_weights` pero nunca se usaban en ningún lado (el comentario decía "ya
   usas el sampler", pero el sampler no existía) — el desbalance de clases no se corregía. Ahora
   `src/car_damage/data.py::make_weighted_sampler` sí se usa en los notebooks 03/04.

**Databricks Free Edition es serverless CPU.** Entrenar el CNN desde cero a 224px por 40 épocas
tomaría horas. Por eso `notebooks/00_config.py` expone `IMG_SIZE` (default 160) y `EPOCHS_CNN`
(default 15) como parámetros ajustables por variable de entorno, en vez de hardcodear los valores
usados en Colab con GPU. Colab sigue siendo el lugar para experimentar libremente; Databricks es
donde el modelo queda gobernado, versionado y servido.

**Las imágenes viven en un Volume, no en Delta.** Delta solo guarda el índice (`path, label,
split, sha256, dims`). Guardar píxeles en Delta solo tiene sentido en datasets de juguete como el
Fashion-MNIST de los notebooks de clase.

**El endpoint recibe base64, no tensores.** El wrapper `CarDamagePyfunc` decodifica la imagen y
aplica el preprocesamiento correcto (CNN: normalización simple; ResNet: estadísticas de
ImageNet) del lado del servidor. Así la webapp no necesita replicar el pipeline de PyTorch.

## Setup

### GitHub → Databricks (CI/CD)

En `Settings → Secrets and variables → Actions` del repo:

| Secret | Descripción |
|---|---|
| `DATABRICKS_HOST` | `https://<workspace>.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | PAT con permisos de Workspace |

Flujo: push a `dev` con un commit que contenga la palabra `PR` → se abre PR automático hacia
`main` (`auto-pr-dev-to-main.yml`). Al mergear a `main` → se sincroniza `notebooks/` y `src/` al
Workspace de Databricks (`deploy-main-to-databricks.yml`). `ci-lint.yml` corre en cada PR:
valida que `src/car_damage` importe y que la webapp compile.

### Databricks

Catálogo: `workspace` (default de Free Edition). Ejecutar en orden los notebooks 01 → 07 (cada
uno documenta sus prerrequisitos en su celda de cabecera).

### Webapp (Vercel)

Variables de entorno del proyecto en Vercel (ver `webapp/.env.example`):

- `DATABRICKS_HOST`, `DATABRICKS_TOKEN`
- `DATABRICKS_ENDPOINT_CNN` = `car-damage-cnn-scratch`
- `DATABRICKS_ENDPOINT_RESNET` = `car-damage-resnet18`

```bash
cd webapp
npm install
npm run dev
```

## Métrica de negocio

Las clases `minor < moderate < severe` son ordinales: confundir `minor` con `severe` es un error
mucho más costoso para una aseguradora que confundir `minor` con `moderate`. Por eso el criterio
de selección de modelo es **Cohen's Kappa cuadrático**, no accuracy (ver `notebooks/05`).
