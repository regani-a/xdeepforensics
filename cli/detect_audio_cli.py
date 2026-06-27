import os
import json
import torch
import numpy as np
import librosa
import argparse
import matplotlib.pyplot as plt
from models.AASIST import Model as AASISTModel

def load_aasist_model(weight_path, config_path, device):
    # 1. Memuat konfigurasi resmi dari file .conf
    with open(config_path, "r") as f:
        config = json.load(f)
    
    class ModelConfig(dict):
        def __getattr__(self, name):
            return self[name]
        def __setattr__(self, name, value):
            self[name] = value

    d_args = ModelConfig(config["model_config"])
    d_args.flag_Fix_zerophase = True 
    
    # Inisialisasi arsitektur berdasarkan manifes konfigurasi
    model = AASISTModel(d_args)
    
    # 2. Memuat bobot latih dengan penanganan ketat state_dict
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    
    # Kupas bungkus state_dict jika tersimpan di dalam key internal (eval_all_best pasca-training)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    # Verifikasi kompatibilitas arsitektur sebelum memuat ke memori
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys) > 0:
        print(f"[*] WARNING (Missing Keys): {len(missing_keys)} layer tidak terisi bobot.")
    if len(unexpected_keys) > 0:
        print(f"[*] WARNING (Unexpected Keys): {len(unexpected_keys)} struktur bobot tidak cocok.")
        
    model = model.to(device)
    model.eval()
    return model

def preprocess_audio(audio_path, target_samples=64600):
    # Memuat audio dengan resampling paksa ke 16kHz (Standar Penyelidikan Audio)
    X, sr = librosa.load(audio_path, sr=16000)
    
    # Penyelarasan durasi sampel audio (Padding/Truncating)
    X_len = X.shape[0]
    if X_len >= target_samples:
        X = X[:target_samples]
    else:
        nb_dup = int(np.ceil(target_samples / X_len))
        X = np.tile(X, nb_dup)[:target_samples]
        
    # KRUSIAL FORENSIK: Normalisasi amplitudo puncak ke rentang [-1.0, 1.0]
    # Tanpa ini, representasi matematis fitur conv akan hancur menghasilkan skor di bawah 1%
    X = X / (np.max(np.abs(X)) + 1e-7)
        
    x_tensor = torch.FloatTensor(X).unsqueeze(0) 
    return x_tensor

def generate_forensic_spectrogram(audio_path, output_img_path):
    """
    Mengekstrak visualisasi Short-Time Fourier Transform (STFT) ke dalam bentuk 
    Spektrogram Frekuensi Logaritmik untuk pemetaan jejak manipulasi (Artifact vocoder).
    """
    X, sr = librosa.load(audio_path, sr=16000)
    
    plt.figure(figsize=(10, 4))
    # Transformasi fourier untuk memecah gelombang suara ke domain frekuensi
    stft_matrix = librosa.amplitude_to_db(np.abs(librosa.stft(X)), ref=np.max)
    
    # Tampilkan visualisasi spektrum warna fourier
    librosa.display.specshow(stft_matrix, sr=sr, x_axis='time', y_axis='log', cmap='jet')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Analisis Spektrogram Forensik: {os.path.basename(audio_path)}")
    plt.xlabel("Durasi (Detik)")
    plt.ylabel("Frekuensi Log (Hz)")
    plt.tight_layout()
    
    # Simpan sebagai file citra bukti digital untuk dilampirkan ke dashboard / berkas pengadilan
    plt.savefig(output_img_path, dpi=300)
    plt.close()
    print(f"[+] Bukti Visual Spektrogram berhasil disimpan: {output_img_path}")

def verify_voice_forensic(audio_path, weight_path, config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Menggunakan perangkat eksekusi: {device}")
    
    # Bangun direktori penyimpanan alat bukti visual jika belum tersedia
    output_dir = "forensic_outputs"
    os.makedirs(output_dir, exist_ok=True)
    spectrogram_file = os.path.join(output_dir, f"{os.path.basename(audio_path)}_spec.png")
    
    try:
        # Pembangkitan komponen bukti visual spektrogram
        generate_forensic_spectrogram(audio_path, spectrogram_file)
        
        # Eksekusi kalkulasi inferensi neural network
        model = load_aasist_model(weight_path, config_path, device)
        x_tensor = preprocess_audio(audio_path).to(device)
        
        with torch.no_grad():
            # Mengikuti standar main.py: ambil elemen kedua sebagai logits klasifikasi
            _, batch_out = model(x_tensor)
            
            # Normalisasi distribusi probabilitas menggunakan Softmax pada logits klasifikasi
            probs = torch.softmax(batch_out, dim=1).cpu().numpy()[0]
            
        # Mengikuti standar arsitektur AASIST resmi:
        # Indeks 0 adalah FAKE (Spoof), Indeks 1 adalah REAL (Bonafide)
        fake_prob = probs[0] * 100
        real_prob = probs[1] * 100
        
        print("\n==========================================")
        print("     LAPORAN FORENSIK MULTIMEDIA AUDIO     ")
        print("==========================================")
        print(f"Nama Berkas Bukti : {os.path.basename(audio_path)}")
        print(f"Skor Otentik (Real): {real_prob:.2f}%")
        print(f"Skor Sintetis (Fake): {fake_prob:.2f}%")
        print("------------------------------------------")
        
        # Penentuan Verdik berbasis ambang batas (Threshold) probabilitas neural
        if fake_prob > 50.0:
            print("VERDIK: TERDETEKSI MANIPULASI SUARA (DEEPFAKE/VC)")
            print("[Catatan] Periksa area frekuensi tinggi pada spektrogram untuk mencari anomali.")
        else:
            print("VERDIK: REKAMAN TERVERIFIKASI OTENTIK (ASLI)")
            print("[Catatan] Harmonisasi struktur formant suara konsisten dan natural.")
        print("==========================================\n")
        
    except Exception as e:
        import traceback
        print(f"[!] Terjadi kesalahan fatal saat analisis forensik audio:")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pemeriksaan Forensik Suara Sintetis SOTA")
    parser.add_argument("-i", "--input", required=True, help="Path ke file audio (.wav)")
    parser.add_argument("-w", "--weights", default="weights/AASIST.pth", help="Path ke file AASIST.pth")
    parser.add_argument("-c", "--config", required=True, help="Path ke file AASIST.conf")
    args = parser.parse_args()
    
    verify_voice_forensic(args.input, args.weights, args.config)

# python detect_audio.py -i "gpt4_11labs_eni_000000000001.wav" -w "weights/AASIST.pth" -c "config/AASIST.conf"