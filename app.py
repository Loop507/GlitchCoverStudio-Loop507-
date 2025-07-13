import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageChops
import hashlib
import time

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("🎧 GlitchCover Studio by Loop507 [Free App]")
st.markdown("Crea una copertina glitch unica basata sul tuo brano. Carica un file audio e genera arte ispirata al suono.")

def get_dimensions(format_type):
    if format_type == "Quadrato (1:1)":
        return (800, 800)
    elif format_type == "Verticale (9:16)":
        return (720, 1280)
    elif format_type == "Orizzontale (16:9)":
        return (1280, 720)
    else:
        return (800, 800)

def analyze_audio_simple(file):
    try:
        file.seek(0)
        audio_bytes = bytearray(file.read())
        file_size = len(audio_bytes)

        hash_obj = hashlib.sha256(audio_bytes[:10000]).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        random.seed(hash_int % 2147483647)
        np.random.seed(hash_int % 2147483647)

        sample_points = [audio_bytes[i % len(audio_bytes)] for i in range(hash_int % 100 + 100)]

        bpm = 60 + (sum(sample_points[:20]) / len(sample_points[:20])) * 120 / 255
        rms = sum(sample_points[20:40]) / len(sample_points[20:40]) / 255 * 0.1
        dominant_freq = (sum(sample_points[40:60]) / len(sample_points[40:60])) * 10000 / 255
        spectral_centroid = (sum(sample_points[60:80]) / len(sample_points[60:80])) * 5000 / 255
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
            "file_hash": hash_int,
            "file_size": file_size
        }

    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None

def apply_glitch_effect(img, seed):
    random.seed(seed)
    img = img.convert("RGB")
    width, height = img.size

    r, g, b = img.split()
    r = ImageChops.offset(r, random.randint(-10, 10), random.randint(-10, 10))
    g = ImageChops.offset(g, random.randint(-10, 10), random.randint(-10, 10))
    b = ImageChops.offset(b, random.randint(-10, 10), random.randint(-10, 10))
    img = Image.merge("RGB", (r, g, b))

    draw = ImageDraw.Draw(img)
    for _ in range(100):
        x = random.randint(0, width)
        y = random.randint(0, height)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.point((x, y), fill=color)

    for y in range(0, height, 3):
        draw.line([(0, y), (width, y)], fill=(random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)))

    small = img.resize((width // 10, height // 10), Image.BILINEAR)
    img = small.resize(img.size, Image.NEAREST)

    return img

def generate_glitch_image(features, seed, size=(800, 800)):
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    width, height = size

    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)

    palettes = {
        "Energetico": [(255, 70, 30), (255, 140, 0), (255, 90, 60)],
        "Calmo": [(30, 90, 160), (60, 120, 130), (0, 160, 130)],
        "Dinamico": [(255, 0, 120), (0, 255, 70), (60, 90, 255), (255, 255, 0)],
        "Equilibrato": [(130, 0, 180), (0, 160, 160), (160, 120, 255)]
    }
    colors = palettes.get(features["emotion"], palettes["Equilibrato"])

    block_size = max(5, int(features["rms"] * 100))
    for x in range(0, width, block_size):
        for y in range(0, height, block_size):
            jitter = random.randint(-30, 30)
            color = tuple(max(0, min(255, c + jitter)) for c in random.choice(colors))
            draw.rectangle([x, y, x + block_size, y + block_size], fill=color)

    line_count = int(features["dynamic_range"] // 1000) + 3
    for _ in range(line_count):
        start_x = random.randint(0, width)
        start_y = 0 if random.random() < 0.5 else height
        end_x = random.randint(0, width)
        end_y = height if start_y == 0 else 0
        draw.line((start_x, start_y, end_x, end_y), fill=random.choice(colors), width=2)

    circle_count = int(features["bpm"] // 10)
    for _ in range(circle_count):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        r = int(features["rms"] * 100) + random.randint(5, 20)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=random.choice(colors), width=2)

    img = apply_glitch_effect(img, seed)

    descrizione = [
        f"🎵 BPM stimato: {features['bpm']:.1f}",
        f"🔊 RMS (energia media): {features['rms']:.3f}",
        f"📡 Frequenza dominante: {features['dominant_freq']:.0f} Hz",
        f"🎼 Centro spettrale: {features['spectral_centroid']:.0f} Hz",
        f"🎭 Stile emotivo: {features['emotion']}"
    ]

    return img, descrizione

if 'regen_count' not in st.session_state:
    st.session_state.regen_count = 0

audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

col1, col2, col3 = st.columns(3)
with col1:
    img_format = st.selectbox("Formato immagine:", ["PNG", "JPEG"])
with col2:
    aspect_ratio = st.selectbox("Formato dimensioni:", ["Quadrato (1:1)", "Verticale (9:16)", "Orizzontale (16:9)"])
with col3:
    show_analysis = st.checkbox("Mostra analisi", value=True)

if audio_file:
    features = analyze_audio_simple(audio_file)
    if features:
        dimensions = get_dimensions(aspect_ratio)
        base_seed = features['file_hash']

        if st.button("🔄 Rigenera Copertina"):
            st.session_state.regen_count += 1

        seed = base_seed + st.session_state.regen_count

        with st.spinner("🎨 Creazione copertina glitch..."):
            img, descrizione = generate_glitch_image(features, seed, size=dimensions)

        st.image(img, caption=f"Copertina glitch generata - {aspect_ratio}", use_container_width=True)

        if show_analysis:
            st.markdown("### 🎨 Descrizione Artistica")
            for d in descrizione:
                st.markdown(d)

        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()

        filename = f"glitch_cover_{features['emotion'].lower()}_{int(time.time())}.{img_format.lower()}"
        st.download_button(
            label=f"⬇️ Scarica {img_format} ({dimensions[0]}x{dimensions[1]})",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}"
        )
else:
    st.session_state.regen_count = 0
    st.info("👆 Carica un file audio per iniziare!")
