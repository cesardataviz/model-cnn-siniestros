"use client";

import { useMemo, useRef, useState } from "react";

type ModelKey = "scratch" | "resnet";
type Severity = "minor" | "moderate" | "severe";
type ImageView = "original" | "gradcam";

interface SelectedImage {
  id: string;
  file: File;
  previewUrl: string;
  base64: string;
}

interface ResultData {
  predictedClass: Severity;
  confidence: number;
  probabilities: Record<string, number>;
  gradcamOverlayBase64: string;
}

interface ResultEntry {
  image: SelectedImage;
  status: "loading" | "done" | "error";
  data?: ResultData;
  error?: string;
}

interface BatchMeta {
  model: ModelKey;
  endpoint: string;
  latencyMs: number;
  imageCount: number;
  requestedAt: Date;
}

const MODEL_LABELS: Record<ModelKey, string> = {
  scratch: "CNN desde cero",
  resnet: "ResNet18 (Transfer Learning)",
};

const BACKBONE_LABELS: Record<ModelKey, string> = {
  scratch: "Entrenado desde cero (sin prior)",
  resnet: "ImageNet-1K (fine-tuned)",
};

const CLASS_ORDER: Severity[] = ["minor", "moderate", "severe"];
const MAX_IMAGES = 10;

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result as string).split(",")[1] ?? "");
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function confidenceTier(confidence: number): "tier-high" | "tier-mid" | "tier-low" {
  if (confidence >= 0.85) return "tier-high";
  if (confidence >= 0.6) return "tier-mid";
  return "tier-low";
}

export default function Home() {
  const [selected, setSelected] = useState<SelectedImage[]>([]);
  const [model, setModel] = useState<ModelKey>("resnet");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ResultEntry[] | null>(null);
  const [meta, setMeta] = useState<BatchMeta | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const summary = useMemo(() => {
    if (!results) return null;
    const counts: Record<Severity, number> = { minor: 0, moderate: 0, severe: 0 };
    let done = 0;
    for (const r of results) {
      if (r.status === "done" && r.data) {
        counts[r.data.predictedClass]++;
        done++;
      }
    }
    return { counts, done, total: results.length };
  }, [results]);

  const selectedEntry = results?.find((r) => r.image.id === selectedId) ?? null;

  async function addFiles(files: FileList | File[]) {
    const imgFiles = Array.from(files).filter((f) => f.type.startsWith("image/"));
    const room = MAX_IMAGES - selected.length;
    const toAdd = imgFiles.slice(0, Math.max(room, 0));

    const newItems: SelectedImage[] = await Promise.all(
      toAdd.map(async (file) => ({
        id: `${file.name}-${file.size}-${crypto.randomUUID()}`,
        file,
        previewUrl: URL.createObjectURL(file),
        base64: await fileToBase64(file),
      }))
    );

    setSelected((prev) => [...prev, ...newItems]);
    setResults(null);
    setMeta(null);
    setError(null);
  }

  function removeImage(id: string) {
    setSelected((prev) => prev.filter((img) => img.id !== id));
  }

  async function handlePredict() {
    if (selected.length === 0) return;
    setLoading(true);
    setError(null);
    setResults(selected.map((image) => ({ image, status: "loading" })));

    const clientStart = performance.now();
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: selected.map((s) => s.base64), model }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Error desconocido");

      const finished = selected.map((image, i) => ({
        image,
        status: "done" as const,
        data: data.results[i],
      }));
      setResults(finished);
      setSelectedId(finished[0]?.image.id ?? null);
      setMeta({
        model,
        endpoint: data.endpoint,
        latencyMs: data.latencyMs ?? Math.round(performance.now() - clientStart),
        imageCount: selected.length,
        requestedAt: new Date(),
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Error al predecir";
      setError(message);
      setResults(selected.map((image) => ({ image, status: "error", error: message })));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <div className="brand-mark">+</div>
            <div className="brand-text">
              <span className="brand-title">Triage de Siniestros</span>
              <span className="brand-subtitle">Clasificador de severidad vehicular</span>
            </div>
          </div>
          <span className="model-pill">{MODEL_LABELS[model]}</span>
        </div>
      </header>

      <main className="page">
        <div className="header">
          <h1>Clasificador de Severidad de Siniestros</h1>
          <p>Sube una o varias fotos del daño vehicular. Cada resultado incluye el mapa Grad-CAM que explica la clasificación.</p>
        </div>

        <div className="card">
          <div className="card-label">Carga &amp; Triage</div>
          <div
            className={`dropzone ${dragOver ? "dragover" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
            }}
          >
            Arrastra fotos aquí o haz clic para subirlas (JPG/PNG, hasta {MAX_IMAGES})
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/png, image/jpeg"
            multiple
            onChange={(e) => e.target.files && addFiles(e.target.files)}
            style={{ display: "none" }}
          />

          {selected.length > 0 && (
            <>
              <div className="thumb-grid">
                {selected.map((img) => (
                  <div className="thumb" key={img.id}>
                    <img src={img.previewUrl} alt={img.file.name} />
                    <button className="thumb-remove" onClick={() => removeImage(img.id)} type="button" aria-label="Quitar">
                      ×
                    </button>
                  </div>
                ))}
              </div>
              <span className="count-badge">{selected.length} / {MAX_IMAGES} imágenes seleccionadas</span>
            </>
          )}

          <div className="model-toggle">
            {(["scratch", "resnet"] as ModelKey[]).map((key) => (
              <button
                key={key}
                className={`model-btn ${model === key ? "active" : ""}`}
                onClick={() => setModel(key)}
                type="button"
              >
                {MODEL_LABELS[key]}
              </button>
            ))}
          </div>

          <button className="predict-btn" onClick={handlePredict} disabled={selected.length === 0 || loading}>
            {loading ? "Clasificando..." : `Clasificar severidad${selected.length > 1 ? ` (${selected.length} fotos)` : ""}`}
          </button>

          {error && <div className="error-box">{error}</div>}
        </div>

        {summary && (
          <div className="summary-row">
            <div className="stat-card">
              <span className="stat-label">Total analizadas</span>
              <span className="stat-value">{summary.done}</span>
            </div>
            <div className="stat-card accent-severe">
              <span className="stat-label">Severe</span>
              <span className="stat-value">{summary.counts.severe}</span>
            </div>
            <div className="stat-card accent-moderate">
              <span className="stat-label">Moderate</span>
              <span className="stat-value">{summary.counts.moderate}</span>
            </div>
            <div className="stat-card accent-minor">
              <span className="stat-label">Minor</span>
              <span className="stat-value">{summary.counts.minor}</span>
            </div>
          </div>
        )}

        {results && (
          <div className="split-layout">
            <div className="result-list">
              {results.map((r) => (
                <ResultRow
                  key={r.image.id}
                  entry={r}
                  active={r.image.id === selectedId}
                  onSelect={() => setSelectedId(r.image.id)}
                />
              ))}
            </div>

            <Inspector entry={selectedEntry} meta={meta} />
          </div>
        )}
      </main>
    </>
  );
}

function ResultRow({
  entry,
  active,
  onSelect,
}: {
  entry: ResultEntry;
  active: boolean;
  onSelect: () => void;
}) {
  const { image, status, data, error } = entry;

  if (status === "loading") {
    return <div className="result-row loading">Clasificando…</div>;
  }
  if (status === "error" || !data) {
    return (
      <div className="result-row errored">
        <strong>{image.file.name}</strong>
        <div>{error || "No se pudo clasificar esta imagen."}</div>
      </div>
    );
  }

  return (
    <button type="button" className={`result-row ${active ? "active" : ""}`} onClick={onSelect}>
      <img className="result-row-thumb" src={image.previewUrl} alt={image.file.name} />
      <div className="result-row-body">
        <span className="result-row-name">{image.file.name}</span>
        <span className={`severity ${data.predictedClass}`}>{data.predictedClass}</span>
      </div>
      <span className={`confidence-value ${confidenceTier(data.confidence)}`}>
        {(data.confidence * 100).toFixed(0)}%
      </span>
    </button>
  );
}

function Inspector({ entry, meta }: { entry: ResultEntry | null; meta: BatchMeta | null }) {
  const [view, setView] = useState<ImageView>("original");

  if (!entry || entry.status !== "done" || !entry.data) {
    return (
      <div className="inspector">
        <span className="inspector-empty">Selecciona una imagen clasificada para ver el detalle.</span>
      </div>
    );
  }

  const { image, data } = entry;

  return (
    <div className="inspector">
      <div className="inspector-header">
        <h3>Visor Forense de Daño</h3>
        <span className="mono-tag">{image.file.name}</span>
      </div>

      <div className="inspector-image-wrap">
        <img
          src={view === "original" ? image.previewUrl : `data:image/png;base64,${data.gradcamOverlayBase64}`}
          alt={view === "original" ? "Foto original" : "Mapa de calor Grad-CAM"}
        />
        <div className="view-toggle">
          <button type="button" className={view === "original" ? "active" : ""} onClick={() => setView("original")}>
            Original
          </button>
          <button type="button" className={view === "gradcam" ? "active" : ""} onClick={() => setView("gradcam")}>
            Grad-CAM
          </button>
        </div>
      </div>

      <div className="diagnosis-row">
        <div>
          <span className="diagnosis-label">Severidad diagnosticada</span>
          <div className="diagnosis-value-row">
            <span className={`severity lg ${data.predictedClass}`}>{data.predictedClass}</span>
            <span className="confidence-value">{(data.confidence * 100).toFixed(1)}% confianza</span>
          </div>
        </div>
        {meta && (
          <div className="diagnosis-arch">
            <span className="diagnosis-label">Arquitectura</span>
            <span className="mono-tag strong">{MODEL_LABELS[meta.model]}</span>
          </div>
        )}
      </div>

      <div className="prob-block">
        <div className="prob-block-heading">
          <span>Clase de daño</span>
          <span>Probabilidad</span>
        </div>
        {CLASS_ORDER.map((cls) => (
          <div className="prob-row" key={cls}>
            <span className="prob-label">{cls}</span>
            <div className="prob-bar-bg">
              <div className="prob-bar-fill" style={{ width: `${(data.probabilities[cls] ?? 0) * 100}%` }} />
            </div>
            <span className="prob-value">{((data.probabilities[cls] ?? 0) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>

      {meta && (
        <div className="telemetry">
          <span className="telemetry-heading">Telemetría del lote</span>
          <div className="telemetry-grid">
            <div className="telemetry-item">
              <span className="telemetry-label">Latencia total</span>
              <span className="telemetry-value">{meta.latencyMs.toLocaleString()} ms</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Latencia / imagen</span>
              <span className="telemetry-value">{Math.round(meta.latencyMs / meta.imageCount).toLocaleString()} ms</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Imágenes en el lote</span>
              <span className="telemetry-value">{meta.imageCount}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Backbone</span>
              <span className="telemetry-value small">{BACKBONE_LABELS[meta.model]}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Endpoint</span>
              <span className="telemetry-value small">{meta.endpoint}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Consulta</span>
              <span className="telemetry-value small">{meta.requestedAt.toLocaleTimeString()}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
