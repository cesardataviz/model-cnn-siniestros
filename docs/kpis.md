# KPIs — antes / después de la implementación de MLOps

La rúbrica pide medir KPIs antes y después. Plantilla a completar con datos reales o supuestos
razonables del caso de negocio (aseguradora) para la sección "Resultados" del informe.

| KPI | Antes (proceso manual) | Después (con el clasificador) | Cómo se mide en este repo |
|---|---|---|---|
| Tiempo de triage por siniestro | Ajustador revisa fotos manualmente (minutos-horas, con cola) | Predicción en segundos vía endpoint REST | Latencia del endpoint (`notebooks/07`, smoke test) |
| Consistencia del criterio | Depende del ajustador (subjetivo) | Mismo modelo, mismo criterio siempre | Reproducibilidad garantizada por versión Delta + versión de modelo en MLflow |
| Costo de error "lejano" (minor↔severe) | No medido explícitamente | Cuantificado con Cohen's Kappa cuadrático | `test_quadratic_kappa` logueado por run (`notebooks/03`, `04`) |
| Trazabilidad (qué datos entrenaron qué modelo) | Manual / inexistente | Automática: cada run de MLflow referencia la versión Delta exacta | `mlflow.log_input` + `DESCRIBE HISTORY` |
| Tiempo de re-entrenamiento ante nuevos datos | Ad-hoc, sin proceso | Notebooks 01-07 encadenados, parametrizados | Pipeline completo, IMG_SIZE/EPOCHS configurables |
| Auditabilidad de decisiones del modelo | Ninguna (caja negra) | Grad-CAM por predicción, loggeado como artefacto | `notebooks/03`/`04`, carpeta `gradcam/` en cada run |

Nota: los valores numéricos de "antes" dependen del caso (real o ficticio) que se documente en la
Introducción del informe -- esta tabla es la estructura, no el dato final.
