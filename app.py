import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import librosa
import io
import random
import soundfile as sf
import os

# ------------------ CONFIGURAZIONE PAGINA ------------------
st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")
st.markdown("""
    <style>
        body { background-color: #0d1117; color: #ffffff; }
        .block-container { padding-top: 2rem; }
        h1, h2, h3, h4 { color: #33ffcc; }
        .stButton>button { background-color: #33ffcc; color: black; border-radius: 0.5rem; padding: 0.5rem 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("GlitchCover Studio by Loop507")
st.subheader("Crea una copertina unica dal tuo brano audio")
st.info("Formato supportato: WAV, MP3 • Max 200MB")

# ------------------ UPLOAD FILE AUDIO ------------------
audio_file = st.file_uploader("🎵 Carica il tuo brano audio", type=["wav", "mp3"])

# ------------------ FUNZIONI ------------------
def analyze_audio(file):
    y, sr = librosa.load(file, sr=None)
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    freqs = librosa.fft_frequencies(sr=sr)
    dominant_freq = freqs[np.argmax(np.abs(np.fft.fft(y)[:len(freqs)]))]
    return tempo, rms, dominant_freq, centroid

def describe_audio(tempo, rms, dominant_freq, centroid):
    mood = "energetico" if tempo > 120 else ("calmo" if tempo < 80 else "moderato")
    description = f"Copertina generata da un brano {mood} con {round(tempo)} BPM.\n"
    description += f"Frequenza dominante: {int(dominant_freq)} Hz\n"
    description += f"Centro spettrale medio: {int(centroid)} Hz\n"
    description += f"Intensità media (RMS): {round(rms, 3)}"
    return description

def generate_glitch_cover(width, height, tempo, rms, freq):
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)
    
    for i in range(100):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = x1 + random.randint(10, 100)
        y2 = y1 + random.randint(5, 50)

        r = int(min(255, tempo * 2 + random.randint(0, 30)))
        g = int(min(255, rms * 10000 % 255))
        b = int(min(255, freq % 255))
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b), outline=None)

    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img

def convert_img(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ------------------ MAIN ------------------
if audio_file is not None:
    with st.spinner("🔍 Analisi audio in corso..."):
        with open("temp_audio", "wb") as f:
            f.write(audio_file.read())
        tempo, rms, dominant_freq, centroid = analyze_audio("temp_audio")
        os.remove("temp_audio")

    st.success("Analisi completata!")
    st.write(f"**BPM stimati:** {round(tempo)}")
    st.write(f"**Intensità RMS:** {round(rms, 3)}")
    st.write(f"**Frequenza dominante:** {int(dominant_freq)} Hz")
    st.write(f"**Centro spettrale:** {int(centroid)} Hz")

    format_option = st.selectbox("📐 Seleziona il formato della copertina", ["Quadrato 1:1", "16:9 Orizzontale", "Verticale 9:16"])
    if format_option == "Quadrato 1:1":
        size = (1024, 1024)
    elif format_option == "16:9 Orizzontale":
        size = (1280, 720)
    else:
        size = (720, 1280)

    st.write("🎨 Generazione della copertina in corso...")
    cover = generate_glitch_cover(size[0], size[1], tempo, rms, dominant_freq)
    cover_data = convert_img(cover)

    st.image(cover, caption="Copertina generata", use_container_width=True)

    st.download_button(
        label="⬇️ Scarica la copertina",
        data=cover_data,
        file_name="glitch_cover.png",
        mime="image/png"
    )

    description = describe_audio(tempo, rms, dominant_freq, centroid)
    st.subheader("📄 Descrizione tecnica della copertina")
    st.text(description)

else:
    st.info("Carica un file audio per iniziare.")
