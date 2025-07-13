import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageFilter
import hashlib
import time
import math

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("🎨 GlitchCover Studio by Loop507")
st.markdown("""
Carica un brano audio per generare una copertina glitch basata sulle sue caratteristiche sonore.

**Limit**: 200MB • Formati supportati: MP3, WAV, OGG
""")

# Analisi Audio

def analyze_audio(file):
    try:
        file.seek(0)
        audio_bytes = file.read()
        if len(audio_bytes) < 10000:
            st.warning("⚠️ File troppo breve. Usa un brano più lungo.")
            return None

        hash_val = int(hashlib.sha256(audio_bytes[:10000]).hexdigest()[:8], 16)
        np.random.seed(hash_val % (2**31))
        random.seed(hash_val)

        chunks = [audio_bytes[i:i+1000] for i in range(0, min(len(audio_bytes), 50000), 1000)]
        freqs = [50 + (sum(chunk)/max(len(chunk),1)/255)*8000 for chunk in chunks if chunk]

        variance = np.var([b for b in audio_bytes[::1000][:100]])
        bpm = 60 + (variance / 255) * 120
        rms = np.mean([b for b in audio_bytes[::5000][:100]]) / 255 * 0.08
        dominant = max(freqs) if freqs else 1000
        centroid = np.mean(freqs) if freqs else 2000
        harmonics = [dominant * i for i in [1,1.5,2] if dominant * i < 8000]
        bass = min(freqs) if freqs else 100
        treble = max(freqs) if freqs else 5000
        dynamic = max(freqs) - min(freqs) if freqs else 1000

        if bpm > 120 and rms > 0.04:
            mood, color = "Energetico", "rosso-arancio"
        elif bpm < 80 and rms < 0.03:
            mood, color = "Calmo", "blu-verde"
        elif dynamic > 2000:
            mood, color = "Dinamico", "multicolore"
        else:
            mood, color = "Equilibrato", "viola-cyan"

        if bpm > 120 and dynamic > 1500:
            genre = "Elettronica/Dance"
        elif bpm < 80 and bass < 200:
            genre = "Acustica/Classica"
        elif dynamic > 2500:
            genre = "Rock/Metal"
        else:
            genre = "Pop/Indie"

        return {
            "bpm": bpm,
            "rms": rms,
            "dominant_freq": dominant,
            "spectral_centroid": centroid,
            "frequencies": freqs,
            "harmonics": harmonics,
            "bass_freq": bass,
            "treble_freq": treble,
            "dynamic_range": dynamic,
            "emotion": mood,
            "emotion_color": color,
            "genre_style": genre,
            "file_hash": hash_val
        }
    except Exception as e:
        st.error(f"Errore: {e}")
        return None

# Formato

def get_size(ratio):
    return {
        "Quadrato (1:1)": (800, 800),
        "Verticale (9:16)": (720, 1280),
        "Orizzontale (16:9)": (1280, 720)
    }.get(ratio, (800, 800))

# Copertina

def generate_cover(fx, seed=0, size=(800,800)):
    width, height = size
    base_colors = {
        "Energetico": [(255,100,0)],
        "Calmo": [(0,150,200)],
        "Dinamico": [(255,0,100),(0,255,100),(100,100,255)],
        "Equilibrato": [(150,0,200)]
    }.get(fx['emotion'], [(100,100,100)])

    random.seed(fx['file_hash'] + seed)
    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)

    for x in range(0, width, 6):
        for y in range(0, height, 6):
            color = random.choice(base_colors)
            intensity = int(fx['dominant_freq'] / 8000 * 100) + random.randint(-10,10)
            pixel = tuple(min(255, int(c * intensity / 100)) for c in color)
            draw.rectangle([x, y, x+6, y+6], fill=pixel)

    return img

# UI
uploaded = st.file_uploader("🎵 Carica brano", type=["mp3","wav","ogg"])

col1, col2 = st.columns(2)
format_opt = col1.selectbox("Formato immagine", ["Quadrato (1:1)","Orizzontale (16:9)","Verticale (9:16)"])
img_fmt = col2.selectbox("Formato file", ["PNG","JPEG"])

if 'seed' not in st.session_state:
    st.session_state.seed = random.randint(0, 999999)

if uploaded:
    fx = analyze_audio(uploaded)
    if fx:
        st.subheader("🎧 Analisi Audio")
        st.metric("BPM", f"{fx['bpm']:.1f}")
        st.metric("Energia RMS", f"{fx['rms']:.3f}")
        st.metric("Dominante", f"{fx['dominant_freq']:.0f} Hz")
        st.metric("Emozione", fx['emotion'])
        st.info(f"🎼 Genere stimato: {fx['genre_style']}")

        st.image(generate_cover(fx, st.session_state.seed, get_size(format_opt)), caption="Copertina generata")

        if st.button("🔁 Rigenera immagine"):
            st.session_state.seed = random.randint(0, 999999)
            st.experimental_rerun()

        # Download
        buf = io.BytesIO()
        image = generate_cover(fx, st.session_state.seed, get_size(format_opt))
        image.save(buf, format=img_fmt)
        st.download_button("⬇️ Scarica Copertina", buf.getvalue(), file_name=f"cover_{int(time.time())}.{img_fmt.lower()}", mime=f"image/{img_fmt.lower()}")
else:
    st.info("👆 Carica un brano per iniziare!")
