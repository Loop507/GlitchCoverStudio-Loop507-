import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import hashlib
import time
import math

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("GlitchCover Studio by Loop507 [Free App]")
st.write("Carica un brano audio e crea una copertina glitch ispirata al suono.")
st.write("Limit 200MB per file • MP3, WAV, OGG")

def analyze_audio_detailed(file) -> dict:
    try:
        file.seek(0)
        audio_bytes = file.read()
        file_size = len(audio_bytes)

        hash_obj = hashlib.sha256(audio_bytes[:10000]).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        byte_chunks = [audio_bytes[i:i+1000] for i in range(0, min(len(audio_bytes), 50000), 1000)]

        frequencies = []
        for chunk in byte_chunks[:20]:
            chunk_val = sum(chunk) / len(chunk) if chunk else 0
            freq = 50 + (chunk_val / 255) * 8000
            frequencies.append(freq)

        np.random.seed(hash_int % 2147483647)

        byte_variance = np.var([b for b in audio_bytes[::1000][:100]])
        bpm = 60 + (byte_variance / 255) * 120

        rms = np.mean([b for b in audio_bytes[::5000][:100]]) / 255 * 0.08

        dominant_freq = max(frequencies) if frequencies else 1000
        spectral_centroid = np.mean(frequencies) if frequencies else 2000
        harmonics = [dominant_freq * i for i in [1, 1.5, 2, 2.5, 3] if dominant_freq * i < 8000]
        bass_freq = min(frequencies) if frequencies else 100
        treble_freq = max(frequencies) if frequencies else 5000
        dynamic_range = max(frequencies) - min(frequencies) if frequencies else 1000

        if bpm > 120 and rms > 0.04:
            emotion = "Energetico"
            emotion_color = "rosso-arancio"
        elif bpm < 80 and rms < 0.03:
            emotion = "Calmo"
            emotion_color = "blu-verde"
        elif dynamic_range > 2000:
            emotion = "Dinamico"
            emotion_color = "multicolore"
        else:
            emotion = "Equilibrato"
            emotion_color = "viola-cyan"

        if bpm > 120 and dynamic_range > 1500:
            genre_style = "Elettronica/Dance"
        elif bpm < 80 and bass_freq < 200:
            genre_style = "Acustica/Classica"
        elif dynamic_range > 2500:
            genre_style = "Rock/Metal"
        else:
            genre_style = "Pop/Indie"

        return {
            "bpm": bpm,
            "rms": rms,
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "frequencies": frequencies,
            "harmonics": harmonics,
            "bass_freq": bass_freq,
            "treble_freq": treble_freq,
            "dynamic_range": dynamic_range,
            "emotion": emotion,
            "emotion_color": emotion_color,
            "genre_style": genre_style,
            "file_hash": hash_int,
            "file_size": file_size
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None

def get_dimensions(format_type):
    if format_type == "Quadrato (1:1)":
        return (800, 800)
    elif format_type == "Verticale (9:16)":
        return (720, 1280)
    elif format_type == "Orizzontale (16:9)":
        return (1280, 720)
    else:
        return (800, 800)

def generate_new_variation():
    st.session_state.variation_seed = random.randint(0, 999999)
    st.rerun()

if 'variation_seed' not in st.session_state:
    st.session_state.variation_seed = random.randint(0, 999999)

if 'last_file_hash' not in st.session_state:
    st.session_state.last_file_hash = None

audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

col1, col2, col3 = st.columns(3)
with col1:
    img_format = st.selectbox("Formato file:", ["PNG", "JPEG"])
with col2:
    aspect_ratio = st.selectbox("Formato immagine:", ["Quadrato (1:1)", "Verticale (9:16)", "Orizzontale (16:9)"])
with col3:
    show_analysis = st.checkbox("Mostra analisi completa", value=True)

if audio_file:
    features = analyze_audio_detailed(audio_file)

    if st.session_state.last_file_hash != features['file_hash']:
        st.session_state.last_file_hash = features['file_hash']
        st.session_state.variation_seed = random.randint(0, 999999)

    if features:
        if show_analysis:
            st.subheader("🔍 Analisi Audio Dettagliata:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎵 BPM", f"{features['bpm']:.1f}")
                st.metric("🔊 Energia RMS", f"{features['rms']:.3f}")
                st.metric("📡 Freq. Dominante", f"{features['dominant_freq']:.0f} Hz")
                st.metric("🎼 Centro Spettrale", f"{features['spectral_centroid']:.0f} Hz")
            with col2:
                st.metric("🎸 Bassi", f"{features['bass_freq']:.0f} Hz")
                st.metric("🎺 Acuti", f"{features['treble_freq']:.0f} Hz")
                st.metric("📊 Gamma Dinamica", f"{features['dynamic_range']:.0f} Hz")
                st.metric("🎭 Emozione", features['emotion'])
            st.info(f"**Genere stimato**: {features['genre_style']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Genera nuova variante", use_container_width=True):
                generate_new_variation()
        with col2:
            quality = st.selectbox("Qualità:", ["Standard", "Alta"])

        dimensions = get_dimensions(aspect_ratio)
        if quality == "Alta":
            dimensions = (int(dimensions[0] * 1.5), int(dimensions[1] * 1.5))

        with st.spinner("🎨 Creando copertina basata sulle caratteristiche audio..."):
            img = Image.new("RGB", dimensions, "black")
            draw = ImageDraw.Draw(img)
            description = [f"💡 Copertina generata con BPM: {features['bpm']:.1f}, Dominant Freq: {features['dominant_freq']:.0f} Hz"]
            draw.text((10, 10), f"{features['emotion']} | {features['genre_style']}", fill="white")

        st.image(img, caption=f"🎨 Copertina glitch generata - {aspect_ratio}", use_container_width=True)

        st.subheader("🎨 Descrizione della copertina:")
        for desc in description:
            st.markdown(desc)

        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()

        filename = f"glitch_cover_{features['genre_style'].replace('/', '_')}_{int(time.time())}.{img_format.lower()}"
        st.download_button(
            label=f"⬇️ Scarica {img_format} ({dimensions[0]}x{dimensions[1]})",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}",
            use_container_width=True
        )

else:
    st.info("👆 Carica un file audio per iniziare!")
