import React, { useEffect, useRef } from "react";

export interface VisualEvidenceItem {
  timestamp: number;
  description: string;
  score?: number;
  coordinates?: { x: number; y: number; w: number; h: number };
  pinpoint?: { id: number; name: string; x: number; y: number };
}

interface GradCamOverlayProps {
  matrix: number[][] | undefined;
  isFake: boolean;
  visualEvidence?: VisualEvidenceItem[];
  currentTime?: number;
}

export default function GradCamOverlay({ 
  matrix, 
  isFake, 
  visualEvidence,
  currentTime = 0 
}: GradCamOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    // Set canvas dimensions based on parent container dynamically
    const resizeCanvas = () => {
      const rect = parent.getBoundingClientRect();
      const width = Math.round(rect.width) || 640;
      const height = Math.round(rect.height) || 360;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    };

    resizeCanvas();
    const resizeObserver = new ResizeObserver(() => {
      resizeCanvas();
    });
    resizeObserver.observe(parent);

    // Find sibling video element if any
    const video = parent.querySelector("video");

    // JET color map mapping helper
    const getJetColor = (v: number) => {
      const val = Math.max(0, Math.min(1, v));
      let r = 0, g = 0, b = 0, a = 0;

      if (val >= 0.75) {
        a = 0.65; // Critical manipulation zone: Red
      } else if (val >= 0.4) {
        a = 0.45; // Transition: Yellow/Orange
      } else if (val >= 0.15) {
        a = 0.25; // Low contribution: Blue/Cyan
      } else {
        a = 0.0; // Authentic: Transparent
      }

      if (a === 0.0) return { r: 0, g: 0, b: 0, a: 0 };

      // JET Map values
      if (val < 0.25) {
        const f = val / 0.25;
        r = 0;
        g = Math.round(255 * f);
        b = 255;
      } else if (val < 0.5) {
        const f = (val - 0.25) / 0.25;
        r = 0;
        g = 255;
        b = Math.round(255 * (1 - f));
      } else if (val < 0.75) {
        const f = (val - 0.5) / 0.25;
        r = Math.round(255 * f);
        g = 255;
        b = 0;
      } else {
        const f = (val - 0.75) / 0.25;
        r = 255;
        g = Math.round(255 * (1 - f));
        b = 0;
      }

      return { r, g, b, a };
    };

    // Main render loop
    const render = () => {
      if (!ctx || !canvas) return;

      const width = canvas.width;
      const height = canvas.height;

      // Clear the canvas
      ctx.clearRect(0, 0, width, height);

      // Only overlay if manipulation is detected
      if (!isFake) {
        animationRef.current = requestAnimationFrame(render);
        return;
      }

      const currentVideoTime = video ? video.currentTime : currentTime;
      const isVideoPlaying = video ? !video.paused : false;
      const time = performance.now();

      // Find the closest active face and body evidence at the exact current time frame
      let activeFaceEv: VisualEvidenceItem | null = null;
      let activeBodyEv: VisualEvidenceItem | null = null;

      if (visualEvidence && visualEvidence.length > 0) {
        let minFaceDiff = Infinity;
        let minBodyDiff = Infinity;

        for (const ev of visualEvidence) {
          const diff = Math.abs(ev.timestamp - currentVideoTime);
          const isBody = ev.description.includes("BODY");

          if (isBody) {
            if (diff < minBodyDiff && diff <= 0.5) {
              minBodyDiff = diff;
              activeBodyEv = ev;
            }
          } else {
            if (diff < minFaceDiff && diff <= 0.3) {
              minFaceDiff = diff;
              activeFaceEv = ev;
            }
          }
        }
      }

      // Helper function to draw circular thermal hotspot around a pinpoint (matches the report visual precisely)
      const drawThermalHotspot = (px: number, py: number, radius: number) => {
        ctx.save();
        ctx.globalCompositeOperation = "screen";
        const grad = ctx.createRadialGradient(px, py, radius * 0.05, px, py, radius);
        // Soft glowing jet-like spectrum
        grad.addColorStop(0.0, "rgba(239, 68, 68, 0.75)");   // Deep Red Alert Center
        grad.addColorStop(0.2, "rgba(249, 115, 22, 0.65)");  // Orange
        grad.addColorStop(0.45, "rgba(234, 179, 8, 0.5)");   // Yellow glow
        grad.addColorStop(0.75, "rgba(34, 197, 94, 0.25)");  // Soft green aura
        grad.addColorStop(0.9, "rgba(59, 130, 246, 0.08)");   // Ambient blue edge
        grad.addColorStop(1.0, "rgba(59, 130, 246, 0.0)");    // Fade out
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.restore();
      };

      // Helper function to draw Grad-CAM matrix heatmaps within clean bounding boxes
      const drawMatrixHeatmap = (targetEv: VisualEvidenceItem) => {
        if (!matrix || matrix.length === 0 || !targetEv.coordinates) return;
        const rows = matrix.length;
        const cols = matrix[0].length;

        let breathe = 0.95 + 0.05 * Math.sin(time * 0.003);
        if (video && isVideoPlaying) {
          breathe = 0.9 + 0.1 * Math.sin(time * 0.008 + currentVideoTime * 4);
        }

        const animatedMatrix = matrix.map((rowArr) =>
          rowArr.map((val) => {
            const wave = 0.02 * Math.sin(time * 0.003);
            const modulated = val * breathe * 1.25 + wave;
            return Math.max(0.0, Math.min(1.0, modulated));
          })
        );

        const interpW = 32;
        const interpH = 32;
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = interpW;
        tempCanvas.height = interpH;
        const tempCtx = tempCanvas.getContext("2d");

        if (tempCtx) {
          const imgData = tempCtx.createImageData(interpW, interpH);
          const data = imgData.data;

          for (let y = 0; y < interpH; y++) {
            for (let x = 0; x < interpW; x++) {
              const gx = (x / (interpW - 1)) * (cols - 1);
              const gy = (y / (interpH - 1)) * (rows - 1);

              const x0 = Math.floor(gx);
              const x1 = Math.min(x0 + 1, cols - 1);
              const y0 = Math.floor(gy);
              const y1 = Math.min(y0 + 1, rows - 1);

              const tx = gx - x0;
              const ty = gy - y0;

              const v00 = animatedMatrix[y0][x0];
              const v10 = animatedMatrix[y0][x1];
              const v01 = animatedMatrix[y1][x0];
              const v11 = animatedMatrix[y1][x1];

              const val = (1 - tx) * (1 - ty) * v00 +
                          tx * (1 - ty) * v10 +
                          (1 - tx) * ty * v01 +
                          tx * ty * v11;

              const color = getJetColor(val);
              const idx = (y * interpW + x) * 4;

              data[idx] = color.r;
              data[idx + 1] = color.g;
              data[idx + 2] = color.b;
              data[idx + 3] = Math.round(color.a * 255);
            }
          }
          tempCtx.putImageData(imgData, 0, 0);

          const { x, y, w, h } = targetEv.coordinates;
          const bx = (x / 100) * width;
          const by = (y / 100) * height;
          const bw = (w / 100) * width;
          const bh = (h / 100) * height;

          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = "high";
          ctx.drawImage(tempCanvas, bx, by, bw, bh);
        }
      };

      // 1. Render Face and Body Grad-CAM Heatmaps
      if (activeFaceEv) {
        drawMatrixHeatmap(activeFaceEv);
      }

      // 2. Render Pinpoint Heatmaps (for both face and body pinpoints, centering on their coordinates)
      if (activeFaceEv && activeFaceEv.pinpoint && activeFaceEv.coordinates) {
        const px = (activeFaceEv.pinpoint.x / 100) * width;
        const py = (activeFaceEv.pinpoint.y / 100) * height;
        const bw = (activeFaceEv.coordinates.w / 100) * width;
        drawThermalHotspot(px, py, bw * 0.45); // Localized glowing pinpoint highlight
      }

      if (activeBodyEv) {
        // Draw body heatmap inside the entire body bounding box to match the reports
        if (activeBodyEv.coordinates) {
          const { x, y, w, h } = activeBodyEv.coordinates;
          const bx = (x / 100) * width;
          const by = (y / 100) * height;
          const bw = (w / 100) * width;
          const bh = (h / 100) * height;

          // Render a gorgeous body-level thermal gradient hotspot inside the bounding box
          const bpx = activeBodyEv.pinpoint ? (activeBodyEv.pinpoint.x / 100) * width : (bx + bw / 2);
          const bpy = activeBodyEv.pinpoint ? (activeBodyEv.pinpoint.y / 100) * height : (by + bh / 2);
          
          // Render body-level thermal overlay centered on the pinpoint
          drawThermalHotspot(bpx, bpy, bw * 0.4);
        }
      }

      // Draw the active bounding boxes and pinpoint markers
      const drawEvidenceUI = (ev: VisualEvidenceItem, isBody: boolean) => {
        if (!ev.coordinates) return;

        const { x, y, w, h } = ev.coordinates;
        const bx = (x / 100) * width;
        const by = (y / 100) * height;
        const bw = (w / 100) * width;
        const bh = (h / 100) * height;

        ctx.strokeStyle = "#ef4444"; // Red alarm border
        ctx.lineWidth = 2.5;
        ctx.strokeRect(bx, by, bw, bh);

        // Warning Label Badge
        const label = isBody ? "BODY RIGIDITY EXCEEDED" : `ARTIFAC SPASIAL (${ev.score?.toFixed(1) || "DEEPFAKE"}%)`;
        ctx.fillStyle = "#ef4444";
        ctx.font = "bold 10px monospace";
        const tw = ctx.measureText(label).width;
        ctx.fillRect(bx, by - 16, tw + 8, 16);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, bx + 4, by - 4);

        // Tactical Pinpoint Crosshair
        if (ev.pinpoint) {
          const px = (ev.pinpoint.x / 100) * width;
          const py = (ev.pinpoint.y / 100) * height;

          ctx.strokeStyle = "#ef4444";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(px, py, 6, 0, 2 * Math.PI);
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(px, py, 14, 0, 2 * Math.PI);
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]); // Reset line dash

          // Draw crosshair ticks
          ctx.beginPath();
          ctx.moveTo(px - 20, py); ctx.lineTo(px - 8, py);
          ctx.moveTo(px + 8, py); ctx.lineTo(px + 20, py);
          ctx.moveTo(px, py - 20); ctx.lineTo(px, py - 8);
          ctx.moveTo(px, py + 8); ctx.lineTo(px, py + 20);
          ctx.stroke();

          // Draw tactical label
          ctx.fillStyle = "#ef4444";
          ctx.font = "9px monospace";
          const pinText = `[P-${ev.pinpoint.id}: ${ev.pinpoint.name.toUpperCase()}]`;
          const ptw = ctx.measureText(pinText).width;
          ctx.fillRect(px - ptw/2 - 4, py + 22, ptw + 8, 14);
          ctx.fillStyle = "#ffffff";
          ctx.fillText(pinText, px - ptw/2, py + 31);
        }
      };

      if (activeFaceEv) {
        drawEvidenceUI(activeFaceEv, false);
      }
      if (activeBodyEv) {
        drawEvidenceUI(activeBodyEv, true);
      }

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      resizeObserver.disconnect();
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [matrix, isFake, visualEvidence, currentTime]);

  return (
    <canvas
      id="gradcam_heatmap_overlay"
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none mix-blend-screen z-[15]"
    />
  );
}
