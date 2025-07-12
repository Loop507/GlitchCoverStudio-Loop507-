import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import librosa
import io
import random
import os

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="wide")

st.title("GlitchCover Studio by Loop507")
st.write("Carica un file audio e genera una copertina glitch unica ispirata al brano.")

# --- Funzioni di analisi audio ---
def analyze_audio(file_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))
        # Spettro frequenze
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S))
        dominant_freq = freqs[np.argmax(np.mean(S, axis=1))]
        freq_range = (freqs[0], freqs[-1])
        
        return {
            "bpm": tempo,
            "rms": rms,
            "spectral_centroid": spectral_centroid,
            "dominant_freq": dominant_freq,
            "freq_range": freq_range,
            "sr": sr
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio: {e}")
        return None

# --- Funzione glitch image generator ---
def generate_glitch_image(audio_features, style="full", width=800, height=800, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    
    # Parametri da audio
    bpm = audio_features["bpm"]
    rms = audio_features["rms"]
    centroid = audio_features["spectral_centroid"]
    dominant = audio_features["dominant_freq"]
    
    # Base color by RMS (volume)
    base_color_val = int(min(max(rms * 4000, 50), 255))
    base_color = (base_color_val, 50, 255 - base_color_val)
    
    # Genera glitch pattern
    for _ in range(150):
        x0 = random.randint(0, width)
        y0 = random.randint(0, height)
        x1 = x0 + random.randint(10, 80)
        y1 = y0 + random.randint(5, 30)
        
        # Colori e trasparenze variano con bpm e frequenze
        color_variation = (
            min(255, int(base_color[0] + bpm) % 255),
            min(255, int(base_color[1] + centroid/10) % 255),
            min(255, int(base_color[2] + dominant/10) % 255),
        )
        
        draw.rectangle([x0, y0, x1, y1], fill=color_variation)
        
        # Distorsione e linee
        if style == "full" or style == "energia":
            # Onde verticali
            for x in range(0, width, 20):
                offset = int(10 * np.sin(x / 20 + bpm / 10))
                draw.line([(x, 0), (x, height)], fill=(base_color_val, 0, 0), width=1)
                
        if style == "full" or style == "tensione":
            # Linee orizzontali glitch
            for y in range(0, height, 15):
                offset = int(5 * np.cos(y / 15 + rms * 100))
                draw.line([(0, y), (width, y + offset)], fill=(0, base_color_val, 0), width=1)
        
    # Blur per effetto artistico
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img

# --- Interfaccia utente ---

uploaded_audio = st.file_uploader("Carica il file audio (max 200MB):", type=["mp3", "wav", "flac"])

img_format = st.selectbox("Scegli il formato immagine:", ["PNG", "JPEG"])

if uploaded_audio:
    # Leggi bytes
    audio_bytes = uploaded_audio.read()
    
    # Analisi audio
    st.info("Analizzo il file audio...")
    features = analyze_audio(audio_bytes)
    
    if features:
        st.success("Analisi completata!")
        st.write(f"🎵 BPM stimati: {features['bpm']:.1f}")
        st.write(f"🔊 Energia (RMS): {features['rms']:.4f}")
        st.write(f"🔎 Frequenza dominante: {features['dominant_freq']:.1f} Hz")
        st.write(f"📡 Centro spettrale medio: {features['spectral_centroid']:.1f} Hz")
        st.write(f"🌈 Gamma frequenze: {features['freq_range'][0]:.1f} Hz – {features['freq_range'][1]:.1f} Hz")
        
        # Bottone rigenera
        if st.button("🎨 Genera copertina glitch"):
            # Usa hash del file per il seed così cambia ogni file
            seed = hash(audio_bytes) % (2**32)
            img = generate_glitch_image(features, style="full", seed=seed)
            
            buf = io.BytesIO()
            img.save(buf, format=img_format)
            byte_im = buf.getvalue()
            
            st.image(img, caption="Copertina glitch generata", use_container_width=True)
            st.download_button("⬇️ Scarica immagine", byte_im, file_name=f"cover_glitch.{img_format.lower()}", mime=f"image/{img_format.lower()}")
            
else:
    st.info("👆 Carica un file audio per iniziare!")

# Footer
st.markdown("---")
st.markdown("GlitchCover Studio by Loop507 - Creazione artistica e tecnica di copertine glitch audio-visive.")
