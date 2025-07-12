import streamlit as st
from PIL import Image
import numpy as np
import io
import random
import librosa
import matplotlib.pyplot as plt

# Stile CSS futuristico e scuro
st.markdown(
    """
    <style>
    .main {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    header, footer {visibility: hidden;}
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        margin-top: 10px;
    }
    .stFileUploader>div>div {
        border: 2px dashed #1f6feb;
        border-radius: 12px;
        padding: 20px;
        background-color: #1a1a1a;
    }
    </style>
    """, unsafe_allow_html=True)

# Titolo e descrizione
st.title("GlitchCover Studio by Loop507 [Free App]")
st.write("🎵 Carica un brano e genera una copertina glitch unica basata sulle sue frequenze audio.")
st.info("Limit 200MB per file • MP3, WAV")

# Caricamento file audio
uploaded_file = st.file_uploader("Carica il file audio", type=["mp3", "wav"])

def analyze_audio(file):
    y, sr = librosa.load(file, sr=None)
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    dominant_freq = centroid  # approssimazione
    bandwidth = np.ptp(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    energy = np.sum(librosa.feature.rms(y=y))
    return {
        "bpm": bpm,
        "rms": rms,
        "dominant_freq": dominant_freq,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "energy": energy
    }

def generate_glitch_image(audio_data, size=(512,512)):
    # Usa dati audio per variare colori e pattern
    base = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Parametri basati sui dati audio (normalizzati)
    bpm_norm = min(audio_data["bpm"]/200, 1.0)
    rms_norm = min(audio_data["rms"]*10, 1.0)
    freq_norm = min(audio_data["dominant_freq"]/10000, 1.0)

    # Genera pattern glitch usando sinusoidi e rumore
    for y in range(size[1]):
        for x in range(size[0]):
            r = int(128 + 127 * np.sin(2 * np.pi * (x / 20 + bpm_norm * y / 5)))
            g = int(128 + 127 * np.sin(2 * np.pi * (y / 15 + rms_norm * x / 7)))
            b = int(128 + 127 * np.sin(2 * np.pi * (x / 25 + freq_norm * y / 3)))
            # Distorsione casuale
            noise = random.randint(-30, 30)
            base[y,x] = [
                np.clip(r + noise, 0, 255),
                np.clip(g + noise, 0, 255),
                np.clip(b + noise, 0, 255)
            ]

    return Image.fromarray(base)

def img_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

if uploaded_file is not None:
    try:
        audio_data = analyze_audio(uploaded_file)
        
        st.subheader("🎶 Dati audio estratti")
        st.write(f"- BPM: {audio_data['bpm']:.1f}")
        st.write(f"- Intensità (RMS): {audio_data['rms']:.4f}")
        st.write(f"- Frequenza dominante: {audio_data['dominant_freq']:.1f} Hz")
        st.write(f"- Centro spettrale medio: {audio_data['centroid']:.1f} Hz")
        st.write(f"- Gamma frequenze (bandwidth): {audio_data['bandwidth']:.1f}")
        
        # Genera immagine glitch
        img = generate_glitch_image(audio_data)
        
        st.subheader("🎨 Copertina generata")
        st.image(img, use_container_width=True)
        
        # Descrizione
        descrizione = (f"L’onda distorta riflette il ritmo elevato di {audio_data['bpm']:.0f} BPM, "
                      f"con frequenze dominanti intorno a {audio_data['dominant_freq']:.0f} Hz e "
                      f"un’intensità RMS di {audio_data['rms']:.3f}.")
        st.markdown(f"*{descrizione}*")
        
        # Download immagine
        img_bytes = img_to_bytes(img)
        st.download_button("⬇️ Scarica la copertina", img_bytes, "glitch_cover.png", "image/png")
        
        # Bottone per rigenerare immagine
        if st.button("🎲 Rigenera la copertina"):
            st.experimental_rerun()

    except Exception as e:
        st.error(f"Errore durante l'elaborazione: {e}")
else:
    st.info("👆 Carica un file audio per iniziare")

# Footer minimale
st.markdown("---")
st.markdown("© Loop507 — GlitchCover Studio")
