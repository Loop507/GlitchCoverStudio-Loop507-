import streamlit as st
import numpy as np
from pydub import AudioSegment
import io
import random
from PIL import Image, ImageDraw
import hashlib
import time
import math

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("🎧 GlitchCover Studio by Loop507 [Free App]")
st.write("Carica un brano audio e genera una copertina glitch ispirata al suono.")
st.write("Limit 200MB • MP3, WAV, OGG")

def get_dimensions(format_type):
    if format_type == "Quadrato (1:1)":
        return (800, 800)
    elif format_type == "Verticale (9:16)":
        return (720, 1280)
    elif format_type == "Orizzontale (16:9)":
        return (1280, 720)
    else:
        return (800, 800)

def analyze_audio(file):
    try:
        # Carica il file audio
        audio = AudioSegment.from_file(file)
        audio = audio.set_channels(1).set_frame_rate(44100)  # mono, 44.1kHz
        samples = np.array(audio.get_array_of_samples())
        frame_rate = audio.frame_rate

        # Calcola RMS (energia)
        rms = np.sqrt(np.mean(samples**2)) / 32768.0  # normalizzato tra 0-1

        # Frequenza dominante tramite FFT ridotta
        fft_size = min(len(samples), 1024)
        fft = np.fft.fft(samples[:fft_size])
        freqs = np.fft.fftfreq(fft_size, 1/frame_rate)
        magnitudes = np.abs(fft[:fft_size//2])
        dominant_freq = freqs[np.argmax(magnitudes[1:]) + 1]  # escludi lo zero

        # Centro spettrale medio
        spectral_centroid = np.sum(freqs[:len(magnitudes)] * magnitudes) / np.sum(magnitudes)

        # BPM simulato (puoi migliorarlo con rilevatore di battiti)
        bpm = 120 + int((rms * 40))

        # Gamma dinamica
        dynamic_range = np.max(magnitudes) - np.min(magnitudes)

        # Emozione e genere stilizzato basati sui valori
        hash_obj = hashlib.sha256(file.getvalue()[:10000]).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        if rms > 0.05 and dominant_freq > 2000:
            emotion = "Energetico"
        elif rms < 0.02 and dominant_freq < 500:
            emotion = "Calmo"
        elif dynamic_range > 1000:
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
            "dominant_freq": abs(dominant_freq),
            "spectral_centroid": abs(spectral_centroid),
            "dynamic_range": abs(dynamic_range),
            "emotion": emotion,
            "genre_style": genre_style,
            "file_hash": hash_int,
            "file_size": file.size
        }

    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None

def generate_glitch_image(features, seed, size=(800, 800)):
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    width, height = size

    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)

    # Palette colori
    if features['emotion'] == "Energetico":
        colors = [(255, 70, 30), (255, 140, 0), (255, 90, 60)]
    elif features['emotion'] == "Calmo":
        colors = [(30, 90, 160), (60, 120, 130), (0, 160, 130)]
    elif features['emotion'] == "Dinamico":
        colors = [(255, 0, 120), (0, 255, 70), (60, 90, 255), (255, 255, 0)]
    else:
        colors = [(130, 0, 180), (0, 160, 160), (160, 120, 255)]

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
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=random.choice(colors), width=2)

    description = [
        f"🎵 BPM stimati: {features['bpm']:.1f}",
        f"🔊 Energia (RMS): {features['rms']:.3f}",
        f"📡 Frequenza dominante: {features['dominant_freq']:.0f} Hz",
        f"🎼 Centro spettrale medio: {features['spectral_centroid']:.0f} Hz",
        f"🎭 Emozione: {features['emotion']}",
        f"🎸 Genere/Stile stimato: {features['genre_style']}"
    ]

    return img, description

# --- UI Streamlit ---
audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

col1, col2, col3 = st.columns(3)
with col1:
    img_format = st.selectbox("Formato immagine:", ["PNG", "JPEG"])
with col2:
    aspect_ratio = st.selectbox("Formato dimensioni:", ["Quadrato (1:1)", "Verticale (9:16)", "Orizzontale (16:9)"])
with col3:
    show_analysis = st.checkbox("Mostra analisi", value=True)

if audio_file:
    features = analyze_audio(audio_file)
    if features:
        st.subheader("🔍 Analisi Audio:")
        if show_analysis:
            st.write(f"BPM stimati: {features['bpm']:.1f}")
            st.write(f"Energia (RMS): {features['rms']:.3f}")
            st.write(f"Frequenza dominante: {features['dominant_freq']:.0f} Hz")
            st.write(f"Centro spettrale medio: {features['spectral_centroid']:.0f} Hz")
            st.write(f"Emozione: {features['emotion']}")
            st.write(f"Genere stimato: {features['genre_style']}")

        dimensions = get_dimensions(aspect_ratio)
        seed = features['file_hash']

        with st.spinner("🎨 Creazione copertina glitch..."):
            img, description = generate_glitch_image(features, seed, size=dimensions)

        st.image(img, caption=f"Copertina glitch generata - {aspect_ratio}", use_container_width=True)

        st.subheader("🎨 Descrizione copertina:")
        for d in description:
            st.markdown(d)

        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()

        filename = f"glitch_cover_{features['genre_style'].replace('/', '_')}_{int(time.time())}.{img_format.lower()}"
        st.download_button(
            label=f"⬇️ Scarica {img_format} ({dimensions[0]}x{dimensions[1]})",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}"
        )

else:
    st.info("👆 Carica un file audio per iniziare!")
