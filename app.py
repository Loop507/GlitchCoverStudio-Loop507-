import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageChops
import hashlib
import time
import librosa

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
        file.seek(0)
        data, sr = librosa.load(file, sr=None, mono=True)
        
        # BPM (tempo)
        tempo, _ = librosa.beat.beat_track(y=data, sr=sr)

        # RMS energy (volume)
        rms = np.mean(librosa.feature.rms(y=data))

        # Spectral centroid (frequenza dominante percepita)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sr))

        # Spectral bandwidth (variazione frequenze)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sr))

        # Frequenza fondamentale stimata con autocorrelazione
        f0, voiced_flag, voiced_probs = librosa.pyin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        fundamental_freq = np.nanmean(f0)
        if np.isnan(fundamental_freq):
            fundamental_freq = spectral_centroid  # fallback

        # Creiamo un hash sul file per seed casuale
        file.seek(0)
        audio_bytes = file.read()
        hash_obj = hashlib.sha256(audio_bytes[:10000]).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        # Determiniamo l'emozione in base a bpm e rms
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

    r, g, b = img.split()
    r = ImageChops.offset(r, random.randint(-15, 15), random.randint(-15, 15))
    g = ImageChops.offset(g, random.randint(-15, 15), random.randint(-15, 15))
    b = ImageChops.offset(b, random.randint(-15, 15), random.randint(-15, 15))
    img = Image.merge("RGB", (r, g, b))

    draw = ImageDraw.Draw(img)
    # pixel glitch casuali
    for _ in range(200):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.point((x, y), fill=color)

    # linee orizzontali sottili glitch
    for y in range(0, height, 2):
        draw.line([(0, y), (width, y)], fill=(random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)))

    # pixelation forte
    small = img.resize((max(1,width // 15), max(1,height // 15)), Image.BILINEAR)
    img = small.resize(img.size, Image.NEAREST)

    return img

def generate_glitch_image(features, seed, size=(800, 800)):
    random.seed(seed)
    width, height = size

    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)

    # Palette basata sull'emozione
    palettes = {
        "Energetico": [(255, 50, 20), (255, 150, 10), (255, 80, 60)],
        "Calmo": [(20, 90, 160), (40, 130, 180), (10, 160, 130)],
        "Dinamico": [(255, 0, 150), (0, 255, 100), (60, 80, 255), (255, 255, 60)],
        "Equilibrato": [(120, 0, 180), (0, 150, 150), (160, 120, 255)]
    }
    colors = palettes.get(features["emotion"], palettes["Equilibrato"])

    # Blocchi colorati che reagiscono a RMS e frequenza fondamentale
    block_size = max(6, int(features["rms"] * 200))
    freq_factor = min(1.0, features["fundamental_freq"] / 4000)
    for x in range(0, width, block_size):
        for y in range(0, height, block_size):
            jitter = random.randint(-40, 40)
            base_color = random.choice(colors)
            # Modifico la luminosità in base alla frequenza fondamentale
            color = tuple(
                max(0, min(255, int(c * freq_factor) + jitter))
                for c in base_color
            )
            draw.rectangle([x, y, x + block_size, y + block_size], fill=color)

    # Linee diagonali glitch in numero proporzionale a bpm
    line_count = max(3, int(features["bpm"] // 30))
    for _ in range(line_count):
        start_x = random.randint(0, width)
        start_y = 0
        end_x = random.randint(0, width)
        end_y = height
        draw.line((start_x, start_y, end_x, end_y), fill=random.choice(colors), width=2)

    # Cerchi glitch con raggio basato su rms e frequenza fondamentale
    circle_count = max(5, int(features["rms"] * 100 * freq_factor))
    for _ in range(circle_count):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        r = int(features["rms"] * 150) + random.randint(5, 30)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=random.choice(colors), width=2)

    img = apply_glitch_effect(img, seed)

    description = [
        f"🎵 BPM: {features['bpm']:.1f}",
        f"🔊 RMS Energy: {features['rms']:.4f}",
        f"📡 Frequenza Fondamentale Stimata: {features['fundamental_freq']:.0f} Hz",
        f"🎼 Centro Spettrale: {features['spectral_centroid']:.0f} Hz",
        f"🌈 Emozione: {features['emotion']}",
    ]

    mood_descriptions = {
        "Energetico": "Energia vibrante, frequenze alte e ritmo sostenuto.",
        "Calmo": "Atmosfera rilassata, toni morbidi e ritmo lento.",
        "Dinamico": "Contrasti marcati e ritmo variabile.",
        "Equilibrato": "Equilibrio armonico tra ritmo e toni."
    }

    mood_desc = mood_descriptions.get(features['emotion'], "Stile audio unico e indefinibile.")

    full_description = {
        "header": f"Glitch Art — {features['emotion']}",
        "mood": mood_desc,
        "details": description
    }

    return img, full_description

# UI
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

        st.markdown("### 🎨 Descrizione Tecnica")
        st.markdown(f"**{description['header']}**")
        st.markdown(description['mood'])
        for d in description['details']:
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
