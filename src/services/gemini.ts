/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface DetectionResult {
  isFake: boolean;
  confidence: number;
  reasoning: string;
  visualEvidence: {
    timestamp: number;
    description: string;
    coordinates?: { x: number; y: number; w: number; h: number };
  }[];
  audioEvidence: {
    startTime: number;
    endTime: number;
    description: string;
    anomalyType: "spectral" | "prosody" | "noise" | "mismatch";
  }[];
  verdict: string;
  status: "MANIPULATED" | "AUTHENTIC";
  confidence_score: number;
  gradcam_matrix: number[][];
  target_layer: string;
  analysis_indonesian: string;
}

/**
 * Mengirim data bukti digital ke Local Pure SOTA Python Engine
 */
export async function analyzeDeepfake(
  fileData: string, // format Base64 dari App.tsx
  mimeType: string,
  isRealtime: boolean = false
): Promise<DetectionResult> {
  try {
    // 1. Konversi Base64 string ke representasi biner Blob asli
    const byteCharacters = atob(fileData);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: mimeType });
    
    // 2. Masukkan ke dalam FormData biner multipart
    const formData = new FormData();
    const extension = mimeType.split("/")[1] || "bin";
    formData.append("file", blob, isRealtime ? `live_stream.jpg` : `evidence_file.${extension}`);

    // 3. Jalankan pemanggilan AJAX Fetch ke endpoint server lokal Python
    const response = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Koneksi server model gagal dengan kode HTTP status: ${response.status}`);
    }

    const sotaData = await response.json();
    const isFake = sotaData.status === "MANIPULATED";

    // 4. Return Object dipetakan sama persis dengan struktur UI internal berkas zip bawaan
    return {
      isFake: isFake,
      confidence: sotaData.confidence_score / 100, // Menyesuaikan bar progress visual UI (skala 0.0 - 1.0)
      confidence_score: sotaData.confidence_score,
      status: sotaData.status,
      target_layer: sotaData.target_layer,
      gradcam_matrix: sotaData.gradcam_matrix,
      analysis_indonesian: sotaData.analysis_indonesian,
      reasoning: sotaData.analysis_indonesian,
      verdict: isFake 
        ? "Media dinyatakan valid mengandung manipulasi sintetis (DEEPFAKE)."
        : "Media dinyatakan asli (AUTHENTIC) berdasarkan parameter ekstraksi fitur model SOTA.",
      visualEvidence: sotaData.visualEvidence,
      audioEvidence: sotaData.audioEvidence
    };
  } catch (error) {
    console.error("SOTA Engine Connection Error:", error);
    throw new Error("Gagal memproses data. Pastikan script python app.py sudah dinyalakan di terminal komputer local kamu.");
  }
}