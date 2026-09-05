"use client";

import { useState } from "react";

type ModelKey = "scratch" | "resnet";

interface PredictResult {
  model: ModelKey;
  predictedClass: "minor" | "moderate" | "severe";
  confidence: number;
  probabilities: Record<string, number>;
}

const MODEL_LABELS: Record<ModelKey, string> = {
  scratch: "CNN desde cero",
  resnet: "ResNet18 (Transfer Learning)",
};

const CLASS_ORDER = ["minor", "moderate", "severe"];

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(",")[1] ?? "");
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default function Home() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [model, setModel] = useState<ModelKey>("resnet");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(file));
    setImageBase64(await fileToBase64(file));
  }

  async function handlePredict() {
    if (!imageBase64) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageBase64, model }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Error desconocido");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error al predecir");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <div className="header">
        <h1>Clasificador de Severidad de Siniestros</h1>
        <p>Sube una foto del daño vehicular y elige qué modelo usar para el triage.</p>
      </div>

      <div className="card">
        <label className="dropzone" htmlFor="file-input">
          {previewUrl ? "Cambiar imagen" : "Haz clic para subir una foto (JPG/PNG)"}
        </label>
        <input
          id="file-input"
          type="file"
          accept="image/png, image/jpeg"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
        {previewUrl && <img src={previewUrl} alt="preview" className="preview" />}

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

        <button
          className="predict-btn"
          onClick={handlePredict}
          disabled={!imageBase64 || loading}
        >
          {loading ? "Clasificando..." : "Clasificar severidad"}
        </button>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <div className="result">
            <span className={`severity ${result.predictedClass}`}>
              {result.predictedClass}
            </span>
            <p style={{ color: "#555", fontSize: "0.9rem" }}>
              Modelo: {MODEL_LABELS[result.model]} · Confianza:{" "}
              {(result.confidence * 100).toFixed(1)}%
            </p>

            {CLASS_ORDER.map((cls) => (
              <div className="prob-row" key={cls}>
                <span className="prob-label">{cls}</span>
                <div className="prob-bar-bg">
                  <div
                    className="prob-bar-fill"
                    style={{ width: `${(result.probabilities[cls] ?? 0) * 100}%` }}
                  />
                </div>
                <span className="prob-value">
                  {((result.probabilities[cls] ?? 0) * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
