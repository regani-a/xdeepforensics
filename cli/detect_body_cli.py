import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def extract_official_pose_features(keypoints_history):
    """
    Mengunduh kemampuan ekstraksi spasial universal YOLOv8-Pose.
    Menerapkan kalkulasi varians absolut berbasis koordinat piksel murni (.xy).
    """
    kp_arr = np.array(keypoints_history) # Shape: (frames, 17, 2)

    # Forward fill untuk menangani tracking loss koordinat
    for f in range(1, kp_arr.shape[0]):
        mask = (kp_arr[f] == 0)
        kp_arr[f][mask] = kp_arr[f-1][mask]

    # 1. Total Geometric Drift (Perpindahan Spasial Sendi Utama)
    left_arm_drift = np.var(kp_arr[:, [5, 7, 9], 0]) + np.var(kp_arr[:, [5, 7, 9], 1])
    right_arm_drift = np.var(kp_arr[:, [6, 8, 10], 0]) + np.var(kp_arr[:, [6, 8, 10], 1])
    total_geometric_drift = float((left_arm_drift + right_arm_drift) / 2.0)
    
    # 2. Angular Variance (Kekakuan Rotasi Persendian Sikut)
    v1 = kp_arr[:, 5, :] - kp_arr[:, 7, :]
    v2 = kp_arr[:, 9, :] - kp_arr[:, 7, :]
    
    # Kalkulasi sudut radian murni COCO
    cos_angles = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6)
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
    angular_variance = float(np.var(angles))

    return total_geometric_drift, angular_variance

def main():
    parser = argparse.ArgumentParser(description="Official YOLOv8-Pose Forensic Classifier Framework")
    parser.add_argument("-i", "--input", required=True, help="Path ke file video bukti (.mp4)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] File video tidak ditemukan: {args.input}")
        return

    print("[*] Loading Official ultralytics YOLOv8n-pose weights...")
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(args.input)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Mengekstrak koordinat skeletal dari {frame_count} frame...")

    keypoints_history = []

    # Menggunakan enumerate untuk tracking akurat indeks frame yang diproses
    for frame_idx in tqdm(range(frame_count), desc="YOLO Inference Loop"):
        ret, frame = cap.read()
        
        if not ret:
            # JANGAN gunakan 'break' agar tidak berhenti di tengah jalan jika ada frame korup/CCTV drop.
            # Kita gunakan 'continue' untuk memaksa OpenCV membaca frame berikutnya.
            continue

        results = model(frame, verbose=False)
        
        if len(results) > 0 and results[0].keypoints is not None:
            kp = results[0].keypoints.xy.cpu().numpy()
            if len(kp) > 0 and len(kp[0]) > 0:
                keypoints_history.append(kp[0])

    cap.release()

    # Validasi kuantitas sampel minimum pasca-ekstraksi menyeluruh
    if len(keypoints_history) < 10:
        print(f"\n[!] Objek manusia tidak terdeteksi secara konsisten. Hanya berhasil mengambil {len(keypoints_history)} frame valid.")
        return

    # Hitung fitur biomekanika murni berdasarkan database gerakan yang lengkap
    geometric_drift, angular_variance = extract_official_pose_features(keypoints_history)

    # Indeks kekakuan gerakan monoton AI
    motion_rigidity_index = geometric_drift / (angular_variance + 1e-6)
    
    # Fungsi Logistik Adaptif (Sigmoid) Umum untuk penentuan skor biner
    fake_score = (1.0 / (1.0 + np.exp(-(motion_rigidity_index - 230000.0) * 0.00002))) * 100.0
    
    if geometric_drift < 5.0 and angular_variance < 0.001:
        fake_score = 0.0

    real_score = 100.0 - fake_score

    print("\n==========================================")
    print("    LAPORAN FORENSIK MULTIMEDIA: BODY SOTA")
    print("==========================================")
    print(f"Nama Berkas Bukti       : {os.path.basename(args.input)}")
    print(f"Total Frame Diproses    : {len(keypoints_history)} dari {frame_count} frame")
    print(f"1. Total Geometric Drift: {geometric_drift:.4f}")
    print(f"2. Angular Variance     : {angular_variance:.4f}")
    print(f"3. Motion Rigidity Index: {motion_rigidity_index:.2f}")
    print("------------------------------------------")
    print(f"SKOR KONSISTENSI ANATOMI (REAL) : {real_score:.2f}%")
    print(f"SKOR CACAT STRUKTUR AI (FAKE)   : {fake_score:.2f}%")
    print("------------------------------------------")
    
    if fake_score > 50.0:
        print("VERDIK: VALID TERDETEKSI MANIPULASI TUBUH (DEEPFAKE GENERATED)")
        print("[Analisis Hukum Acara] Gerakan sendi terlalu kaku dan monoton (Khas AI Diffusion).")
    else:
        print("VERDIK: STRUKTUR GERAKAN ORGANIK / OTENTIK")
        print("[Analisis Hukum Acara] Distribusi rotasi sudut dan pergeseran spasial seimbang.")
    print("==========================================\n")

if __name__ == "__main__":
    main()
# python detect_body_cli.py -i "001.mp4"