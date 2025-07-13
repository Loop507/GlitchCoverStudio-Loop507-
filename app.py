import streamlit as st
import numpy as np
import io
import random
import hashlib
import time
from PIL import Image, ImageDraw, ImageOps, ImageFilter

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")
st.title("🎧 GlitchCover Studio by Loop507 [Free App]")
st.markdown("Crea una copertina glitch artistica basata sul tuo audio. Carica un file MP3, WAV o OGG.")

# --- Funzioni ---
def get_dimensions(format_type):
    return {
        "Quadrato (1:1)": (800, 800),
        "Verticale (9:16)": (720, 1280),
        "Orizzontale (16:9)": (1280, 720)
    }.get(format_type, (800, 800))

def analyze_audio_simple(file):
    try:
        file.seek(0)
        audio_bytes = bytearray(file.read())
        file_size = len(audio_bytes)

        hash_obj = hashlib.sha256(audio_bytes[:10000]).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        random.seed(hash_int)
        np.random.seed(hash_int)

        sample_points = [audio_bytes[i % len(audio_bytes)] for i in range(100)]

        bpm = 60 + (sum(sample_points[:20]) / 20) * 120 / 255
        rms = sum(sample_points[20:40]) / 20 / 255 * 0.1
        dominant_freq = (sum(sample_points[40:60]) / 20) * 10000 / 255
        spectral_centroid = (sum(sample_points[60:80]) / 20) * 5000 / 255
        dynamic_range = abs(dominant_freq - spectral_centroid)

        if rms > 0.05 and dominant_freq > 2000:
            emotion = "Energetico"
        elif rms < 0.02 and dominant_freq < 500:
            emotion = "Calmo"
        elif dynamic_range > 2000:
            emotion = "Dinamico"
        else:
            emotion = "Equilibrato"

        return {
            "bpm": bpm,
            "rms": rms,
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "dynamic_range": dynamic_range,
            "emotion": emotion,
            "file_hash": hash_int
        }

    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None

def apply_glitch_effect(img, seed):
    random.seed(seed)
    width, height = img.size

    r, g, b = img.split()
    r = ImageOps.offset(r, random.randint(-10, 10), random.randint(-10, 10))
    g = ImageOps.offset(g, random.randint(-10, 10), random.randint(-10, 10))
    b = ImageOps.offset(b, random.randint(-10, 10), random.randint(-10, 10))
    img = Image.merge("RGB", (r, g, b))

    draw = ImageDraw.Draw(img)
    for _ in range(80):
        x = random.randint(0, width)
        y = random.randint(0, height)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.point((x, y), fill=color)

    for y in range(0, height, 3):
        draw.line([(0, y), (width, y)], fill=(random.randint(0, 50),) * 3)

    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img

def generate_glitch_image(features, seed, size):
    width, height = size
    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)

    palette = {
        "Energetico": [(255, 70, 30), (255, 140, 0)],
        "Calmo": [(30, 90, 160), (60, 120, 130)],
        "Dinamico": [(255, 0, 120), (0, 255, 70)],
        "Equilibrato": [(130, 0, 180), (0, 160, 160)]
    }[features["emotion"]]

    for x in range(0, width, 20):
        for y in range(0, height, 20):
            color = random.choice(palette)
            draw.rectangle([x, y, x+20, y+20], fill=color)

    img = apply_glitch_effect(img, seed)

    descrizione = [
        f"🎵 BPM stimato: {features['bpm']:.1f}",
        f"🔊 Intensità RMS: {features['rms']:.3f}",
        f"📡 Frequenza dominante: {features['dominant_freq']:.0f} Hz",
        f"🎼 Centro spettrale: {features['spectral_centroid']:.0f} Hz",
        f"📊 Gamma dinamica: {features['dynamic_range']:.0f} Hz",
        f"🎭 Emozione dominante: {features['emotion']}"
    ]
    return img, descrizione

# --- UI ---
if 'regen_count' not in st.session_state:
    st.session_state.regen_count = 0

audio_file = st.file_uploader("Carica il tuo brano (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])
col1, col2 = st.columns(2)
with col1:
    img_format = st.selectbox("Formato immagine", ["PNG", "JPEG"])
with col2:
    aspect_ratio = st.selectbox("Dimensione immagine", ["Quadrato (1:1)", "Verticale (9:16)", "Orizzontale (16:9)"])

if audio_file:
    features = analyze_audio_simple(audio_file)
    if features:
        dimensions = get_dimensions(aspect_ratio)
        base_seed = features['file_hash']

        if st.button("🔄 Rigenera"):
            st.session_state.regen_count += 1

        seed = base_seed + st.session_state.regen_count

        with st.spinner("Generazione immagine glitch..."):
            img, descrizione = generate_glitch_image(features, seed, size=dimensions)

        st.image(img, caption="Copertina Glitch Generata", use_container_width=True)

        st.markdown("### Descrizione tecnica")
        for d in descrizione:
            st.markdown(d)

        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()
        filename = f"glitch_cover_{int(time.time())}.{img_format.lower()}"

        st.download_button(
            label="⬇️ Scarica",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}"
        )
else:
    st.session_state.regen_count = 0
    st.info("👆 Carica un file audio per iniziare.")
