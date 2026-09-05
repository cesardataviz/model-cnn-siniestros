import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

type ModelKey = "scratch" | "resnet";

const ENDPOINT_BY_MODEL: Record<ModelKey, string | undefined> = {
  scratch: process.env.DATABRICKS_ENDPOINT_CNN,
  resnet: process.env.DATABRICKS_ENDPOINT_RESNET,
};

interface PredictRequestBody {
  imageBase64: string;
  model: ModelKey;
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

  const { imageBase64, model } = body;
  if (!imageBase64 || (model !== "scratch" && model !== "resnet")) {
    return NextResponse.json(
      { error: "Se requiere 'imageBase64' y 'model' ('scratch' | 'resnet')." },
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

  const dbResponse = await fetch(invocationUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ dataframe_records: [{ image_base64: imageBase64 }] }),
  });

  if (!dbResponse.ok) {
    const errorText = await dbResponse.text();
    return NextResponse.json(
      { error: `Databricks respondió ${dbResponse.status}: ${errorText}` },
      { status: 502 }
    );
  }

  const data = await dbResponse.json();
  const prediction = data.predictions?.[0];

  if (!prediction) {
    return NextResponse.json(
      { error: "Respuesta del endpoint sin predicciones." },
      { status: 502 }
    );
  }

  return NextResponse.json({
    model,
    predictedClass: prediction.predicted_class,
    confidence: prediction.confidence,
    probabilities: prediction.probabilities,
  });
}
