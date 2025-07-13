# GlitchCover Studio by Loop507 - Versione Artistica Avanzata

import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import hashlib
import time
import math
from scipy.io import wavfile
from pydub import AudioSegment

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("GlitchCover Studio by Loop507 [Free App]")
st.write("Carica un brano audio e crea una copertina glitch ispirata ai dati del suono.")
st.write("Limit 200MB per file • MP3, WAV, OGG")

# Analisi audio basata su bytes + estrazione più realistica da pydub

def analyze_audio_detailed(file) -> dict:
    try:
        audio = AudioSegment.from_file(file)
        samples = np.array(audio.get_array_of_samples())
        samples = samples.astype(np.float32) / (2 ** (8 * audio.sample_width - 1))

        file.seek(0)
        audio_bytes = file.read()
        file_size = len(audio_bytes)
        hash_obj = hashlib.sha256(audio_bytes[:10000]).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        np.random.seed(hash_int % 2147483647)

        # Calcolo BPM simulato basato su variazione dei campioni
        diff = np.abs(np.diff(samples))
        energy = np.mean(diff)
        bpm = 60 + energy * 240

        rms = np.sqrt(np.mean(samples**2)) * 0.1

        # Frequenza dominante (semplificata)
        spectrum = np.abs(np.fft.fft(samples[:44100]))
        freqs = np.fft.fftfreq(len(spectrum), 1.0 / audio.frame_rate)
        peak_idx = np.argmax(spectrum[:len(spectrum)//2])
        dominant_freq = abs(freqs[peak_idx])

        spectral_centroid = np.sum(freqs[:len(spectrum)//2] * spectrum[:len(spectrum)//2]) / np.sum(spectrum[:len(spectrum)//2])

        harmonics = [dominant_freq * r for r in [1, 1.5, 2] if dominant_freq * r < 8000]
        bass_freq = np.min(freqs[np.where(freqs > 20)])
        treble_freq = np.max(freqs[np.where(freqs < 15000)])
        dynamic_range = np.max(samples) - np.min(samples)

        if bpm > 120 and rms > 0.04:
            emotion = "Energetico"
        elif bpm < 80 and rms < 0.03:
            emotion = "Calmo"
        elif dynamic_range > 1:
            emotion = "Dinamico"
        else:
            emotion = "Equilibrato"

        if bpm > 120 and dynamic_range > 0.5:
            genre_style = "Elettronica/Dance"
        elif bpm < 80:
            genre_style = "Acustica/Classica"
        elif dynamic_range > 1.5:
            genre_style = "Rock/Metal"
        else:
            genre_style = "Pop/Indie"

        return {
            "bpm": bpm,
            "rms": rms,
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "harmonics": harmonics,
            "bass_freq": bass_freq,
            "treble_freq": treble_freq,
            "dynamic_range": dynamic_range,
            "emotion": emotion,
            "genre_style": genre_style,
            "file_hash": hash_int,
            "file_size": file_size
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None

def generate_image(features, size=(800,800)):
    img = Image.new("RGB", size, color="black")
    draw = ImageDraw.Draw(img)
    seed = features['file_hash']
    random.seed(seed)
    np.random.seed(seed % 2147483647)

    base_colors = [(255,0,100), (0,255,150), (150,100,255)]
    particles = int(features['bpm'])
    waves = len(features['harmonics'])
    blocks = int(features['rms'] * 1000)

    for _ in range(particles):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        color = random.choice(base_colors)
        draw.ellipse([x, y, x+3, y+3], fill=color)

    for h in features['harmonics']:
        radius = int(h / 100)
        cx, cy = random.randint(100, size[0]-100), random.randint(100, size[1]-100)
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], outline=random.choice(base_colors))

    for _ in range(blocks):
        x = random.randint(0, size[0]-20)
        y = random.randint(0, size[1]-20)
        draw.rectangle([x, y, x+10, y+10], fill=random.choice(base_colors))

    return img, [
        f"🎵 **BPM**: {features['bpm']:.1f}",
        f"📡 **Dominant Frequency**: {features['dominant_freq']:.1f} Hz",
        f"💥 **Energia RMS**: {features['rms']:.3f}",
        f"🎭 **Emozione**: {features['emotion']}",
        f"🎧 **Genere stimato**: {features['genre_style']}"
    ]

# UI
file = st.file_uploader("🎵 Carica file audio (mp3/wav/ogg)", type=["mp3", "wav", "ogg"])
if file:
    features = analyze_audio_detailed(file)
    if features:
        dim = st.selectbox("Formato immagine", ["Quadrato", "Verticale", "Orizzontale"])
        w, h = (800,800) if dim=="Quadrato" else (720,1280) if dim=="Verticale" else (1280,720)
        img, desc = generate_image(features, size=(w,h))
        st.image(img, caption="🎨 Copertina generata", use_container_width=True)
        st.subheader("🧠 Analisi del brano:")
        for d in desc:
            st.markdown(d)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button("⬇️ Scarica copertina", data=buf.getvalue(), file_name="glitchcover.png", mime="image/png")
else:
    st.info("Carica un file audio per iniziare")
