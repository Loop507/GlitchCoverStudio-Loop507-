import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageFilter
import hashlib
import time
import math

# Impostazioni pagina
st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("GlitchCover Studio by Loop507 🎨🎵")
st.write("Carica un brano audio per generare una copertina artistica unica basata sulle sue caratteristiche sonore.")
st.write("Formato supportato: MP3, WAV, OGG. Dimensione massima: 200MB")

# Analisi simulata dei dati audio da bytes (no librerie audio)
def analyze_audio(file):
    file.seek(0)
    audio_bytes = file.read()
    hash_obj = hashlib.sha256(audio_bytes[:10000]).hexdigest()
    seed = int(hash_obj[:8], 16)
    np.random.seed(seed % (2**32 - 1))
    
    frequencies = np.random.uniform(60, 12000, 20)
    bpm = np.random.uniform(60, 180)
    rms = np.random.uniform(0.01, 0.09)
    dominant_freq = np.median(frequencies)
    spectral_centroid = np.mean(frequencies)
    dynamic_range = np.max(frequencies) - np.min(frequencies)

    if bpm > 130:
        emotion = "Energetico"
        color_palette = [(255, 80, 0), (255, 200, 0)]
    elif bpm < 80:
        emotion = "Intimo"
        color_palette = [(50, 100, 180), (0, 180, 120)]
    else:
        emotion = "Equilibrato"
        color_palette = [(200, 100, 255), (100, 255, 200)]

    return {
        "frequencies": frequencies,
        "bpm": bpm,
        "rms": rms,
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "dynamic_range": dynamic_range,
        "emotion": emotion,
        "color_palette": color_palette,
        "hash": seed
    }

# Genera immagine artistica

def generate_art_cover(data, size=(800, 800), variation=0):
    random.seed(data['hash'] + variation)
    np.random.seed((data['hash'] + variation) % (2**32 - 1))
    img = Image.new("RGB", size, color="black")
    draw = ImageDraw.Draw(img)

    # Fondo glitchato a bande
    for y in range(0, size[1], 10):
        color = random.choice(data['color_palette'])
        glitch_intensity = int(data['dominant_freq'] / 12000 * 255)
        mod_color = tuple(min(255, int(c + random.uniform(-30, 30))) for c in color)
        draw.rectangle([0, y, size[0], y + 5], fill=mod_color)

    # Onde curve
    for f in data['frequencies'][:6]:
        amp = int(f / 1500)
        step = int(800 / 40)
        for x in range(0, size[0], step):
            y = int(size[1]/2 + amp * math.sin(f * x * 0.0001))
            radius = int(data['rms'] * 400 + 2)
            color = random.choice(data['color_palette'])
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)

    # Particelle dinamiche
    for _ in range(int(data['bpm'])):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        color = random.choice(data['color_palette'])
        r = random.randint(1, 3)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

    # Cerchi concentrici armonici
    center = (size[0]//2, size[1]//2)
    for i in range(3):
        r = int((data['spectral_centroid']/12000) * size[0]//2) + i*10
        color = data['color_palette'][i % len(data['color_palette'])]
        draw.ellipse([center[0]-r, center[1]-r, center[0]+r, center[1]+r], outline=color, width=2)

    return img

# Interfaccia
uploaded = st.file_uploader("Carica brano audio", type=["mp3", "wav", "ogg"])

if uploaded:
    data = analyze_audio(uploaded)
    variation = st.slider("Variazione artistica", 0, 20, 0)
    quality = st.selectbox("Formato immagine", ["Standard (800x800)", "Alta (1200x1200)"])
    size = (800, 800) if "Standard" in quality else (1200, 1200)

    with st.spinner("Generazione in corso..."):
        image = generate_art_cover(data, size=size, variation=variation)

    st.image(image, caption=f"Copertina generata - Emozione: {data['emotion']} - BPM: {data['bpm']:.1f}")

    st.markdown("---")
    st.subheader("🎧 Descrizione Artistica della Copertina")
    st.markdown(f"- L'immagine è influenzata dalla **frequenza dominante** di circa **{data['dominant_freq']:.0f} Hz**.")
    st.markdown(f"- Il **battito per minuto (BPM)** di **{data['bpm']:.1f}** ha determinato la quantità e il movimento delle particelle.")
    st.markdown(f"- Le curve glitch sono mappate sulle **frequenze medie** tra **{int(np.min(data['frequencies']))} Hz** e **{int(np.max(data['frequencies']))} Hz**.")
    st.markdown(f"- L'**emozione prevalente** è: **{data['emotion']}**, rappresentata dai colori selezionati.")

    # Download
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    st.download_button("⬇️ Scarica Copertina", buffer.getvalue(), file_name="glitch_cover.png", mime="image/png")
else:
    st.info("Carica un file audio per iniziare la creazione della tua copertina glitch personalizzata.")
