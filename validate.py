#!/usr/bin/env python3
"""
XDEEPFORENSICS AUTOMATED MULTI-MODAL DATASET VALIDATOR (V7 - High Fidelity Metrics)
Mengintegrasikan TestClient FastAPI untuk kalkulasi metrik saintifik: Accuracy, Precision, 
Recall, F1-Score, AUC, EER, dan t-DCF dengan luaran struktur JSON yang rapi.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
from fastapi.testclient import TestClient

# Sembunyikan warning bawaan library agar log konsol tetap bersih dan rapi
warnings.filterwarnings("ignore")

# Sinkronisasi direktori kerja aktif ke sys.path agar modul lokal terbaca
sys.path.append(os.getcwd())

try:
    # Mengimpor aplikasi FastAPI langsung dari core engine main.py
    from main import app
    INTEGRATION_OK = True
except ImportError as e:
    print(f"[!] Gagal mengintegrasikan core dari main.py: {e}")
    print("[!] Pastikan file 'main.py' berada di direktori aktif saat ini (CWD).")
    INTEGRATION_OK = False

# ====================================================================
#                          KONFIGURASI DATASET
# ====================================================================
DATASET_DIR = "/content/drive/MyDrive/Colab Notebooks/xdeepforensics/dataset_validasi"
OUTPUT_REPORT = "laporan_validasi_clean.json"

# Pemetaan otomatis direktori berdasarkan subfolder modalitas barang bukti
TARGET_MAP = {
    "audio_only": {"authentic": "AUTHENTIC", "manipulated": "MANIPULATED"},
    "video_mute": {"authentic": "AUTHENTIC", "manipulated": "MANIPULATED"},
    "video_audio": {"authentic": "AUTHENTIC", "manipulated": "MANIPULATED"},
    "c2pa_validation": {"authentic": "AUTHENTIC", "manipulated": "MANIPULATED"}
}

def run_advanced_dataset_validation():
    print("====================================================================")
    print("    XDEEPFORENSICS AUTOMATED MULTI-MODAL DATASET VALIDATOR (V7)   ")
    print("====================================================================")
    print(f"[*] Target Folder : {DATASET_DIR}")
    print(f"[*] Output Repos  : {OUTPUT_REPORT}")
    
    if not INTEGRATION_OK:
        print("[X] Validasi dibatalkan karena kegagalan integrasi dengan main.py.")
        return

    start_time = time.time()
    report_details = {}
    global_stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "Total_Data": 0}
    
    # Array koleksi data untuk Advanced Metrics (ROC/AUC, EER, t-DCF)
    y_true = []   # Ground Truth biner (1: MANIPULATED, 0: AUTHENTIC)
    y_scores = [] # Probabilitas kontinu kelas MANIPULATED (0.0 - 1.0)

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Folder dataset '{DATASET_DIR}' tidak ditemukan! Periksa path Anda.")
        return

    # Inisialisasi TestClient lokal (berjalan di memori tanpa uvicorn port)
    client = TestClient(app)

    # Menggunakan context manager lifespan agar model SOTA termuat sempurna ke hardware
    with client:
        print("\n[*] Menghubungi FastAPI Lifespan Engine...")
        print("[*] MEMULAI PROSES PENDAHULUAN: Memuat Neural Model SOTA ke memori...")

        # Loop pengecekan status kesiapan model neural backend
        models_ready = False
        for attempt in range(1, 61):  # Maksimal 60 kali percobaan (2 menit)
            try:
                response = client.get("/status")
                if response.status_code == 200:
                    status_data = response.json()
                    if status_data.get("models_ready", False):
                        device = status_data.get("device", "CPU")
                        print(f"[✓] Neural Engine Siap! Hardware Target Acceleration: \033[94m{device}\033[0m")
                        models_ready = True
                        break
            except Exception:
                pass
            if attempt % 5 == 1:
                print(f"    -> [Status Load Model] Menunggu inisialisasi neural weights (Upaya {attempt})...")
            time.sleep(2.0)

        if not models_ready:
            print("[WARNING] Neural Engine tidak merespon status siap. Melanjutkan dengan mode fallback.")

        print("-" * 80)

        # ----------------------------------------------------------------
        # PEMPROSESAN BATCH DATASET BERDASARKAN STRUKTUR SUBFOLDER
        # ----------------------------------------------------------------
        for category_folder, label_map in TARGET_MAP.items():
            category_path = os.path.join(DATASET_DIR, category_folder)
            if not os.path.exists(category_path):
                continue

            print(f"\n\033[1m▶ Memproses Kategori Modalitas: {category_folder}\033[0m")
            report_details[category_folder] = []

            for status_sub, ground_truth in label_map.items():
                sub_folder_path = os.path.join(category_path, status_sub)
                if not os.path.exists(sub_folder_path):
                    continue

                # Filter berkas tersembunyi seperti .DS_Store atau .ipynb_checkpoints
                file_list = [f for f in os.listdir(sub_folder_path) if not f.startswith('.')]
                if not file_list:
                    continue

                print(f"  |- Mengevaluasi {len(file_list)} berkas {status_sub.upper()} (Ground Truth: {ground_truth})")

                for file_name in file_list:
                    file_path = os.path.join(sub_folder_path, file_name)
                    print(f"     [*] Menyeleksi barang bukti: {file_name}...", end="")

                    try:
                        # Kirim payload berkas langsung via HTTP Request mock internal
                        with open(file_path, "rb") as f:
                            response = client.post(
                                "/analyze", 
                                files={"file": (file_name, f, "application/octet-stream")}
                            )

                        if response.status_code == 200:
                            result = response.json()
                            prediction = result.get("status", "ERROR")
                            confidence = result.get("confidence_score", 0.0)
                            
                            # Ekstraksi metadata skor mentah untuk isolasi modalitas
                            meta_data = result.get("_meta", {})
                            raw_face = meta_data.get("face_score_raw", 0.0)
                            raw_audio = meta_data.get("audio_score_raw", 0.0)
                            raw_body = meta_data.get("body_score_raw", 0.0)
                            
                            # --- 1. ISOLASI SKOR RAW BERDASARKAN KLASIFIKASI SUBFOLDER ---
                            if category_folder == "audio_only":
                                face_score, body_score = 0.0, 0.0
                                audio_score = raw_audio
                                selected_raw_score = audio_score
                            elif category_folder == "video_mute":
                                face_score = raw_face
                                audio_score = 0.0
                                body_score = raw_body
                                selected_raw_score = max(face_score, body_score)
                            else:
                                face_score = raw_face
                                audio_score = raw_audio
                                body_score = raw_body
                                selected_raw_score = max(face_score, audio_score, body_score)

                            # --- 2. PENYELARASAN LINEAR PROBABILITAS UNTUK ROC/AUC ---
                            if prediction == "MANIPULATED":
                                final_prob_manipulated = confidence / 100.0 if confidence >= 50.0 else (100.0 - confidence) / 100.0
                            else:
                                final_prob_manipulated = (100.0 - confidence) / 100.0 if confidence >= 50.0 else confidence / 100.0

                        else:
                            prediction, confidence = "ERROR", 0.0
                            face_score, audio_score, body_score, selected_raw_score = 0.0, 0.0, 0.0, 0.0
                            final_prob_manipulated = 0.5

                        # Evaluasi ketepatan prediksi klasifikasi biner
                        is_correct = (ground_truth == prediction) or (ground_truth in ["AUTHENTIC", "ORIGINAL"] and prediction in ["AUTHENTIC", "ORIGINAL"])

                        # Perbarui Matriks Konfusi Global
                        if ground_truth == "MANIPULATED":
                            if is_correct: global_stats["TP"] += 1
                            else: global_stats["FN"] += 1
                        else:
                            if is_correct: global_stats["TN"] += 1
                            else: global_stats["FP"] += 1

                        global_stats["Total_Data"] += 1
                        
                        # Gabungkan ke array statistik global
                        actual_binary = 1 if ground_truth == "MANIPULATED" else 0
                        y_true.append(actual_binary)
                        y_scores.append(final_prob_manipulated)

                        # Cetak status log ke konsol dengan pewarnaan ANSI terstruktur
                        status_color = "\033[92m✓ COCOK\033[0m" if is_correct else "\033[91mX MISSED\033[0m"
                        pred_tag = f"[\033[91m{prediction}\033[0m]" if prediction == "MANIPULATED" else f"[\033[92m{prediction}\033[0m]"
                        print(f"\r     {status_color} | {file_name} -> Prediksi: {pred_tag} Confidence: {confidence:.2f}%")

                        # Rekam entri detail berkas ke dalam repositori struktur JSON laporan
                        report_details[category_folder].append({
                            "file_name": file_name,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                            "confidence_score": round(confidence, 2),
                            "is_correct": is_correct,
                            "_raw_scores_extracted": {
                                "face_raw": round(face_score, 4),
                                "audio_raw": round(audio_score, 4),
                                "body_raw": round(body_score, 4),
                                "filtered_selected_raw": round(selected_raw_score, 4)
                            }
                        })

                    except Exception as ex:
                        print(f"\r     \033[91m[ERROR]\033[0m Gagal memproses berkas {file_name}: {ex}")

    # ----------------------------------------------------------------
    #                      KOMPUTASI AKHIR METRIK
    # ----------------------------------------------------------------
    print("\n[*] Menghitung matriks performa dan metrik saintifik tingkat tinggi...")
    tp, tn, fp, fn = global_stats["TP"], global_stats["TN"], global_stats["FP"], global_stats["FN"]
    total = global_stats["Total_Data"]

    accuracy = ((tp + tn) / total * 100) if total > 0 else 0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # Kalkulasi advanced metrics menggunakan modul eksternal scikit-learn
    try:
        from sklearn.metrics import roc_curve, auc
        y_true_np = np.array(y_true)
        y_scores_np = np.array(y_scores)

        # Hitung Area Under Curve (AUC)
        fpr, tpr, thresholds = roc_curve(y_true_np, y_scores_np)
        roc_auc = auc(fpr, tpr)

        # Hitung Equal Error Rate (EER)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute(fpr - fnr))]

        # Hitung tandem Detection Cost Function (t-DCF) standar kompetisi ASVspoof
        C_miss, C_fa = 1.0, 10.0
        P_tar, P_non = 0.05, 0.95
        t_dcf_array = (C_miss * P_tar * fnr) + (C_fa * P_non * fpr)
        t_dcf = float(np.min(t_dcf_array))
    except Exception as e:
        print(f"[!] Gagal mengalkulasi modul metrik sklearn: {e}")
        roc_auc, eer, t_dcf = 0.0, 0.0, 0.0

    # Menyusun dokumen laporan akhir secara rapi dan hierarkis
    final_report = {
        "execution_info": {
            "total_duration_seconds": round(time.time() - start_time, 2),
            "engine": "Colab Hardware SOTA Live Evaluation"
        },
        "global_metrics": {
            "Accuracy": round(accuracy, 2),
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1_Score": round(f1_score, 2),
            "AUC": round(roc_auc * 100, 2),
            "EER": round(eer * 100, 2),
            "t-DCF": round(t_dcf, 4)
        },
        "global_stats": global_stats,
        "details": report_details
    }

    # Menyimpan file laporan JSON dengan format indentasi renggang yang rapi (Pretty Printed)
    try:
        with open(OUTPUT_REPORT, "w", encoding="utf-8") as json_file:
            json.dump(final_report, json_file, indent=4, ensure_ascii=False)
        
        print("\n====================================================================")
        print("             RINGKASAN AKHIR VALIDASI KLASIFIKASI DATASET           ")
        print("====================================================================")
        print(f"[-] Total Berkas Diuji   : {total} Berkas")
        print(f"[-] Benar Positif (TP)   : {tp} | Benar Negatif (TN) : {tn}")
        print(f"[-] Salah Positif (FP)   : {fp} | Salah Negatif (FN) : {fn}")
        print("-" * 68)
        print(f"[✓] Akurasi Sistem       : {final_report['global_metrics']['Accuracy']}%")
        print(f"[✓] F1-Score Sistem      : {final_report['global_metrics']['F1_Score']}%")
        print(f"[✓] AUC Sistem (ROC)     : {final_report['global_metrics']['AUC']}%")
        print(f"[✓] EER Sistem           : {final_report['global_metrics']['EER']}%")
        print(f"[✓] Tandem DCF Score     : {final_report['global_metrics']['t-DCF']}")
        print("====================================================================")
        print(f"[+] Laporan biner rapi berhasil diproduksi di: {OUTPUT_REPORT}")
    except Exception as e:
        print(f"[X] Gagal menulis log laporan JSON eksternal: {e}")


if __name__ == "__main__":
    # Injeksi runtime patch pengaman module PyTorch sebelum loop berjalan
    import torch
    torch.nn.Module.dump_patches = True
    
    run_advanced_dataset_validation()