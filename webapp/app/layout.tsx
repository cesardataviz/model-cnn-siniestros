import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Triage de Siniestros — Clasificador de Severidad",
  description: "Clasificación de severidad de daños vehiculares (minor / moderate / severe)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
