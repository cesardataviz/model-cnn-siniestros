import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

type ModelKey = "scratch" | "resnet";

const ENDPOINT_BY_MODEL: Record<ModelKey, string | undefined> = {
  scratch: process.env.DATABRICKS_ENDPOINT_CNN,
  resnet: process.env.DATABRICKS_ENDPOINT_RESNET,
};

const MAX_IMAGES_PER_REQUEST = 10;

interface PredictRequestBody {
  images: string[];
  model: ModelKey;
}

interface DatabricksPrediction {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  gradcam_overlay_base64: string;
}

export async function POST(req: NextRequest) {
  const host = process.env.DATABRICKS_HOST;
  const token = process.env.DATABRICKS_TOKEN;

  if (!host || !token) {
    return NextResponse.json(
      { error: "Servidor mal configurado: falta DATABRICKS_HOST o DATABRICKS_TOKEN." },
      { status: 500 }
    );
  }

  let body: PredictRequestBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Body inválido, se espera JSON." }, { status: 400 });
  }

  const { images, model } = body;
  if (!Array.isArray(images) || images.length === 0 || (model !== "scratch" && model !== "resnet")) {
    return NextResponse.json(
      { error: "Se requiere 'images' (arreglo no vacío) y 'model' ('scratch' | 'resnet')." },
      { status: 400 }
    );
  }
  if (images.length > MAX_IMAGES_PER_REQUEST) {
    return NextResponse.json(
      { error: `Máximo ${MAX_IMAGES_PER_REQUEST} imágenes por lote.` },
      { status: 400 }
    );
  }

  const endpointName = ENDPOINT_BY_MODEL[model];
  if (!endpointName) {
    return NextResponse.json(
      { error: `Falta configurar el endpoint para el modelo '${model}'.` },
      { status: 500 }
    );
  }

  const invocationUrl = `${host.replace(/\/$/, "")}/serving-endpoints/${endpointName}/invocations`;

  // Una sola llamada al endpoint con todas las imágenes del lote -- el modelo servido
  // ya procesa un arreglo de dataframe_records, no hace falta invocar una vez por imagen.
  const dbResponse = await fetch(invocationUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      dataframe_records: images.map((imageBase64) => ({ image_base64: imageBase64 })),
    }),
  });

  if (!dbResponse.ok) {
    const errorText = await dbResponse.text();
    return NextResponse.json(
      { error: `Databricks respondió ${dbResponse.status}: ${errorText}` },
      { status: 502 }
    );
  }

  const data = await dbResponse.json();
  const predictions: DatabricksPrediction[] = data.predictions ?? [];

  if (predictions.length !== images.length) {
    return NextResponse.json(
      { error: "El endpoint devolvió un número de resultados distinto al de imágenes enviadas." },
      { status: 502 }
    );
  }

  return NextResponse.json({
    model,
    results: predictions.map((p) => ({
      predictedClass: p.predicted_class,
      confidence: p.confidence,
      probabilities: p.probabilities,
      gradcamOverlayBase64: p.gradcam_overlay_base64,
    })),
  });
}
