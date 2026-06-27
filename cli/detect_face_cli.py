import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image as pil_image
from tqdm import tqdm

# Import InsightFace bawaan dari dokumentasi target
import insightface
from insightface.app import FaceAnalysis

# Modul lokal dari repo FaceForensics++ Anda
from network.models import model_selection, TransferModel
from dataset.transform import xception_default_data_transforms

torch.serialization.add_safe_globals([TransferModel])

def align_and_crop_face(frame, face, target_size=299, scale=1.3):
    """
    Melakukan 5-Point Affine Transformation berdasarkan landmark InsightFace.
    Memastikan koordinat mata dan wajah selalu tegak lurus (Identik dengan cara training FF++).
    """
    # Mengambil koordinat kotak pembatas (bbox)
    bbox = face.bbox.astype(np.int32)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    w = x2 - x1
    h = y2 - y1
    size_bb = int(max(w, h) * scale)
    
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Ambil koordinat titik potong baru
    nx1 = max(int(center_x - size_bb // 2), 0)
    ny1 = max(int(center_y - size_bb // 2), 0)
    
    cropped = frame[ny1:ny1+size_bb, nx1:nx1+size_bb]
    if cropped.size == 0:
        return None
        
    # Ekstraksi Landmark 5 Titik Utama untuk Alinyemen Spasial (Mata, Hidung, Sudut Mulut)
    # Catatan: Karena Anda memuat module 'detection', face.kps berisi 5 keypoints dasar
    if hasattr(face, 'kps'):
        src_pts = face.kps.astype(np.float32)
        
        # Referensi koordinat standar wajah ideal (wajah tegak lurus)
        dst_pts = np.array([
            [30.2946 / 112.0 * size_bb, 51.6963 / 112.0 * size_bb], # Mata Kiri
            [65.5318 / 112.0 * size_bb, 51.5014 / 112.0 * size_bb], # Mata Kanan
            [48.0252 / 112.0 * size_bb, 71.7366 / 112.0 * size_bb], # Hidung
            [33.5493 / 112.0 * size_bb, 92.3655 / 112.0 * size_bb], # Sudut Mulut Kiri
            [62.7299 / 112.0 * size_bb, 92.2041 / 112.0 * size_bb]  # Sudut Mulut Kanan
        ], dtype=np.float32)
        
        # Hitung matriks transformasi afin untuk memutar wajah yang miring
        M, _ = cv2.estimateAffinePartial2D(src_pts - np.array([nx1, ny1]), dst_pts)
        if M is not None:
            aligned_face = cv2.warpAffine(cropped, M, (size_bb, size_bb))
            return aligned_face

    return cv2.resize(cropped, (size_bb, size_bb))

def preprocess_image(image, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0)
    return preprocessed_image.to(device)

def main():
    parser = argparse.ArgumentParser(description="Forensik Wajah SOTA Menggunakan InsightFace Alignment Pipeline")
    parser.add_argument("-i", "--input", required=True, help="Path ke file video")
    parser.add_argument("-mi", "--model_path", default="weights/all_c23.p", help="Path ke weights XceptionNet")
    parser.add_argument("-cf", "--consecutive_frames", type=int, default=5, help="Threshold runtutan frame bising")
    args = parser.parse_args()

    # Paksa penggunaan CUDA jika tersedia demi performa real-time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    print(f"[*] Perangkat Neural Network: {device}")
    print(f"[*] Provider InsightFace: {providers}")

    # 1. Inisialisasi Detektor Ter-align SOTA (RetinaFace)
    print("[*] Menginisialisasi RetinaFace melalui InsightFace Engine...")
    face_app = FaceAnalysis(allowed_modules=['detection'], providers=providers)
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

    # 2. Memuat XceptionNet Classifier
    print("[*] Memuat model klasifikasi spasial XceptionNet...")
    model = model_selection(modelname='xception', num_out_classes=2)
    model_weights = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(model_weights, dict):
        model.load_state_dict(model_weights)
    else:
        model = model_weights
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[!] File video tidak valid: {args.input}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Memulai scanning forensik pada {frame_count} frame...")

    fake_scores = []
    current_consecutive = 0
    max_consecutive = 0
    trigger_threshold = 50.0

    with torch.no_grad():
        for _ in tqdm(range(frame_count), desc="Memproses Frame"):
            ret, frame = cap.read()
            if not ret: break

            # Deteksi wajah menggunakan RetinaFace
            faces = face_app.get(frame)

            if len(faces) > 0:
                # Ambil wajah utama dengan tingkat keyakinan (bbox score) tertinggi
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                face = faces[0]
                
                # Proses alignment wajah secara geometris sebelum inferensi
                aligned_face = align_and_crop_face(frame, face, target_size=299)

                if aligned_face is not None and aligned_face.size > 0:
                    input_tensor = preprocess_image(aligned_face, device)
                    output = torch.softmax(model(input_tensor), dim=1)
                    
                    face_fake_prob = float(output[0][1].cpu().numpy()) * 100.0
                    fake_scores.append(face_fake_prob)

                    # Update runtutan frame anomali
                    if face_fake_prob > trigger_threshold:
                        current_consecutive += 1
                        if current_consecutive > max_consecutive:
                            max_consecutive = current_consecutive
                    else:
                        current_consecutive = 0

    cap.release()

    if len(fake_scores) == 0:
        print("\n[!] Gagal: Tidak ada struktur wajah yang berhasil di-align.")
        return

    # Hitung nilai puncak persentil ke-95 untuk meredam pencilan tunggal (single outlier frame)
    peak_fake_score = np.percentile(fake_scores, 95)
    is_manipulated = False
    verdik_reason = ""

    if max_consecutive >= args.consecutive_frames:
        is_manipulated = True
        verdik_reason = f"Ditemukan glitch manipulasi beruntun sebanyak {max_consecutive} frame."
    elif peak_fake_score > 50.0:
        is_manipulated = True
        verdik_reason = f"Skor anomali spasial pada area teralign kritis ({peak_fake_score:.2f}%)."
    else:
        verdik_reason = "Seluruh struktur piksel wajah konsisten dengan distribusi data kamera alami."

    final_fake_display = peak_fake_score if is_manipulated else np.mean(fake_scores)
    final_real_display = 100.0 - final_fake_display

    print("\n==========================================")
    print("   LAPORAN FORENSIK RETINAFACE-XCEPTION   ")
    print("==========================================")
    print(f"Nama Berkas Bukti : {os.path.basename(args.input)}")
    print(f"Skor Puncak Sintetis (Fake): {final_fake_display:.2f}%")
    print(f"Skor Puncak Otentik (Real): {final_real_display:.2f}%")
    print(f"Runtutan Glitch Maksimal   : {max_consecutive} frames")
    print("------------------------------------------")
    
    if is_manipulated:
        print("VERDIK: TERDETEKSI MANIPULASI WAJAH (DEEPFAKE/FACE-SWAP)")
        print(f"[Alasan] {verdik_reason}")
    else:
        print("VERDIK: REKAMAN WAJAH TERVERIFIKASI OTENTIK (ASLI)")
        print(f"[Alasan] {verdik_reason}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
# python detect_face_cli.py -i "001.mp4" -mi "weights/all_c23.p"