import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageOps
import hashlib
import time
import librosa
import tempfile

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

def analyze_audio_real(file) -> dict:
    try:
        # Salvo temporaneamente il file per lettura sicura con librosa
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(file.read())
            tmp.flush()
            data, sr = librosa.load(tmp.name, sr=None, mono=True)

            tempo, _ = librosa.beat.beat_track(y=data, sr=sr)
            rms = np.mean(librosa.feature.rms(y=data))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sr))
            f0, voiced_flag, voiced_probs = librosa.pyin(data, 
                                                         fmin=librosa.note_to_hz('C2'), 
                                                         fmax=librosa.note_to_hz('C7'))
            fundamental_freq = np.nanmean(f0)
            if np.isnan(fundamental_freq):
                fundamental_freq = spectral_centroid

            # Hash file
            tmp.seek(0)
            audio_bytes = tmp.read()
            hash_obj = hashlib.sha256(audio_bytes[:10000]).hexdigest()
            hash_int = int(hash_obj[:8], 16)

            if tempo > 120 and rms > 0.05:
                emotion = "Energetico"
            elif tempo < 80 and rms < 0.02:
                emotion = "Calmo"
            elif spectral_bandwidth > 4000:
                emotion = "Dinamico"
            else:
                emotion = "Equilibrato"

            return {
                "bpm": tempo,
                "rms": rms,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "fundamental_freq": fundamental_freq,
                "emotion": emotion,
                "file_hash": hash_int,
                "file_size": len(audio_bytes)
            }

    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None

def apply_glitch_effect(img, seed):
    random.seed(seed)
    img = img.convert("RGB")
    width, height = img.size

    from PIL import ImageChops

    r, g, b = img.split()
    r = ImageChops.offset(r, random.randint(-10, 10), random.randint(-10, 10))
    g = ImageChops.offset(g, random.randint(-10, 10), random.randint(-10, 10))
    b = ImageChops.offset(b, random.randint(-10, 10), random.randint(-10, 10))
    img = Image.merge("RGB", (r, g, b))

    draw = ImageDraw.Draw(img)
    for _ in range(100):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.point((x, y), fill=color)

    for y in range(0, height, 3):
        draw.line([(0, y), (width, y)], fill=(random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)))

    small = img.resize((max(1, width // 10), max(1, height // 10)), Image.BILINEAR)
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

    line_count = int(features["spectral_bandwidth"] // 1000) + 3
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

    description = [
        f"🎵 Emozione rilevata: {features['emotion']}",
        f"🔊 Energia RMS: {features['rms']:.3f}",
        f"📡 Frequenza fondamentale stimata: {features['fundamental_freq']:.0f} Hz",
        f"🎼 Centro spettrale: {features['spectral_centroid']:.0f} Hz",
        f"📊 Larghezza di banda spettrale: {features['spectral_bandwidth']:.0f} Hz",
        f"⏱️ BPM stimati: {features['bpm']:.1f}"
    ]

    return img, description

# --- UI ---
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
    features = analyze_audio_real(audio_file)
    if features:
        dimensions = get_dimensions(aspect_ratio)
        base_seed = features['file_hash']

        if st.button("🔄 Rigenera Copertina"):
            st.session_state.regen_count += 1

        seed = base_seed + st.session_state.regen_count

        with st.spinner("🎨 Creazione copertina glitch..."):
            img, description = generate_glitch_image(features, seed, size=dimensions)

        st.image(img, caption=f"Copertina glitch generata - {aspect_ratio}", use_container_width=True)

        if show_analysis:
            st.markdown("### 🎨 Descrizione Tecnica")
            for d in description:
                st.markdown(d)

        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()

        filename = f"glitch_cover_{int(time.time())}.{img_format.lower()}"
        st.download_button(
            label=f"⬇️ Scarica {img_format} ({dimensions[0]}x{dimensions[1]})",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}"
        )

else:
    st.session_state.regen_count = 0
    st.info("👆 Carica un file audio per iniziare!")
