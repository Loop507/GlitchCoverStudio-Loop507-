import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageOps
import hashlib
import time
import math

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("🎧 GlitchCover Studio by Loop507 [Free App]")
st.markdown("Crea una copertina glitch unica basata sul tuo brano. Carica un file audio e genera arte ispirata al suono.")

# Dimensioni immagine

def get_dimensions(format_type):
    return {
        "Quadrato (1:1)": (800, 800),
        "Verticale (9:16)": (720, 1280),
        "Orizzontale (16:9)": (1280, 720)
    }.get(format_type, (800, 800))

# Analisi audio semplificata dai byte

def analyze_audio_simple(file) -> dict:
    file.seek(0)
    audio_bytes = file.read()
    hash_int = int(hashlib.sha256(audio_bytes[:10000]).hexdigest()[:8], 16)
    random.seed(hash_int)
    np.random.seed(hash_int)

    # Sample simulati per dati
    chunk = audio_bytes[:100] if len(audio_bytes) >= 100 else audio_bytes
    vals = [b for b in chunk]

    bpm = 60 + (np.var(vals) / 255) * 120
    rms = (np.mean(vals) / 255) * 0.1
    dominant_freq = (np.mean(vals[:20]) / 255) * 10000
    spectral_centroid = (np.mean(vals[20:40]) / 255) * 5000
    dynamic_range = abs(dominant_freq - spectral_centroid)

    # Stime emozione e genere
    if bpm > 120 and rms > 0.05:
        emotion = "Energetico"
    elif bpm < 80 and rms < 0.02:
        emotion = "Calmo"
    elif dynamic_range > 2000:
        emotion = "Dinamico"
    else:
        emotion = "Equilibrato"

    if dominant_freq > 2000 and bpm > 110:
        genre_style = "Elettronica/Dance"
    elif dominant_freq < 500 and bpm < 80:
        genre_style = "Acustica/Classica"
    elif dynamic_range > 2000:
        genre_style = "Rock/Metal"
    else:
        genre_style = "Pop/Indie"

    return {
        "bpm": bpm,
        "rms": rms,
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "dynamic_range": dynamic_range,
        "emotion": emotion,
        "genre_style": genre_style,
        "file_hash": hash_int
    }

# Glitch image generator avanzato

def generate_glitch_image(features, seed, size=(800,800)):
    random.seed(seed)
    np.random.seed(seed)
    width, height = size
    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)

    # Palette in base all'emozione
    palettes = {
        "Energetico": [(255,70,30),(255,140,0),(255,90,60)],
        "Calmo": [(30,90,160),(60,120,130),(0,160,130)],
        "Dinamico": [(255,0,120),(0,255,70),(60,90,255),(255,255,0)],
        "Equilibrato": [(130,0,180),(0,160,160),(160,120,255)]
    }
    colors = palettes[features["emotion"]]

    # Blocchi colorati
    block = max(5, int(features["rms"] * 100))
    for x in range(0, width, block):
        for y in range(0, height, block):
            c = random.choice(colors)
            jitter = random.randint(-20,20)
            color = tuple(max(0,min(255,v+jitter)) for v in c)
            draw.rectangle([x,y,x+block,y+block], fill=color)

    # Onde sinusoidali
    for i in range(0, width, 10):
        amp = int(features["dynamic_range"] / 100)
        y = int(height/2 + amp * math.sin(i / 50))
        draw.line([(i, y), (i+10, y)], fill=random.choice(colors), width=2)

    # Cerchi in base a BPM
    for _ in range(int(features["bpm"]//10)):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        r = random.randint(5, 20)
        draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline=random.choice(colors), width=2)

    # Effetto finale glitch
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img

# UI Streamlit

audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3","wav","ogg"])
col1,col2,col3 = st.columns(3)
with col1:
    img_format = st.selectbox("Formato file:", ["PNG","JPEG"])
with col2:
    aspect_ratio = st.selectbox("Formato immagine:", ["Quadrato (1:1)","Verticale (9:16)","Orizzontale (16:9)"])
with col3:
    show_analysis = st.checkbox("Mostra analisi tecnica", value=True)

if audio_file:
    features = analyze_audio_simple(audio_file)
    dimensions = get_dimensions(aspect_ratio)
    seed = features["file_hash"]

    if show_analysis:
        st.subheader("🔍 Analisi Audio")
        st.write(f"BPM: {features['bpm']:.1f}")
        st.write(f"RMS: {features['rms']:.3f}")
        st.write(f"Dominant Freq: {features['dominant_freq']:.0f} Hz")
        st.write(f"Spectral Centroid: {features['spectral_centroid']:.0f} Hz")
        st.write(f"Dynamic Range: {features['dynamic_range']:.0f}")
        st.write(f"Emozione: {features['emotion']}")
        st.write(f"Genere/Stile: {features['genre_style']}")

    # Generazione immagine avanzata
    with st.spinner("🎨 Generazione copertina glitch..."):
        img = generate_glitch_image(features, seed, size=dimensions)

    st.image(img, caption=f"Copertina glitch - {aspect_ratio}", use_container_width=True)

    # Descrizione tecnica avanzata
    st.subheader("📄 Descrizione Tecnica")
    st.markdown(f"- **BPM**: {features['bpm']:.1f}")
    st.markdown(f"- **Energia (RMS)**: {features['rms']:.3f}")
    st.markdown(f"- **Frequenza Dominante**: {features['dominant_freq']:.0f} Hz")
    st.markdown(f"- **Centro Spettrale**: {features['spectral_centroid']:.0f} Hz")
    st.markdown(f"- **Gamma Dinamica**: {features['dynamic_range']:.0f}")
    st.markdown(f"- **Emozione**: {features['emotion']}")
    st.markdown(f"- **Genere/Stile**: {features['genre_style']}")

    # Download
    buf = io.BytesIO()
    img.save(buf, format=img_format)
    st.download_button(
        label=f"⬇️ Scarica ({img_format})",
        data=buf.getvalue(),
        file_name=f"glitch_cover_{int(time.time())}.{img_format.lower()}",
        mime=f"image/{img_format.lower()}"
    )
else:
    st.info("👆 Carica un file audio per iniziare!")
