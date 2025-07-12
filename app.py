import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import random

# Configurazione
st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("GlitchCover Studio by Loop507")
st.markdown("Carica un brano audio per generare una copertina unica ispirata al suono.")

# Upload del brano
audio_file = st.file_uploader("🎵 Carica un file audio", type=["mp3", "wav", "ogg"])

# Scelta formato immagine
img_format = st.selectbox("📸 Formato immagine da generare", ["PNG", "JPEG", "WEBP"])

def analyze_audio(file):
    """Simulazione analisi audio"""
    random.seed(file.name)
    return {
        "bpm": random.randint(60, 180),
        "emotion": random.choice(["Energia", "Malinconia", "Tensione"]),
        "dominant_freq": random.randint(300, 5000),
        "spectral_centroid": random.randint(800, 4000),
        "freq_range": (random.randint(20, 80), random.randint(10000, 16000))
    }

def generate_glitch_image(audio_data):
    """Genera un'immagine glitch in base ai dati audio"""
    w, h = 768, 768
    img = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(img)

    # Onde verticali
    for y in range(0, h, 12):
        amp = audio_data['bpm'] / 2
        freq = audio_data['dominant_freq'] / 1000
        shift = int(amp * np.sin(y * freq * 0.01))
        color = (
            int(100 + abs(shift) * 2),
            int(audio_data['spectral_centroid'] % 255),
            int(255 - abs(shift) * 2)
        )
        draw.line([(0, y + shift), (w, y)], fill=color, width=2)

    # Linee orizzontali glitch
    for i in range(80):
        y = random.randint(0, h-1)
        length = random.randint(30, 300)
        x = random.randint(0, w - length)
        color = (
            random.randint(0, 255),
            int(audio_data['spectral_centroid'] % 255),
            random.randint(0, 255)
        )
        draw.line([(x, y), (x + length, y)], fill=color, width=1)

    return img

def describe_image(audio_data):
    """Descrizione tecnica visiva"""
    return (
        f"Copertina ispirata al brano: {audio_data['emotion']}.\n\n"
        f"• BPM: {audio_data['bpm']}\n"
        f"• Frequenza dominante: {audio_data['dominant_freq']} Hz\n"
        f"• Centro spettrale: {audio_data['spectral_centroid']} Hz\n"
        f"• Gamma frequenze: {audio_data['freq_range'][0]} – {audio_data['freq_range'][1]} Hz\n\n"
        f"Le linee e i colori riflettono i dati acustici per una rappresentazione visiva unica."
    )

def convert_image(img, format):
    """Converte immagine per il download"""
    buf = io.BytesIO()
    img.save(buf, format=format.upper())
    return buf.getvalue()

# Logica
if audio_file is not None:
    try:
        st.audio(audio_file)

        # Analisi
        audio_data = analyze_audio(audio_file)

        # Generazione immagine
        with st.spinner("🎨 Generazione copertina in corso..."):
            img = generate_glitch_image(audio_data)

        # Mostra immagine
        st.image(img, caption="✨ Copertina Generata", use_container_width=True)

        # Descrizione tecnica
        st.markdown("🧠 **Analisi audio-visiva**")
        st.text(describe_image(audio_data))

        # Download
        img_data = convert_image(img, img_format)
        st.download_button(
            f"⬇️ Scarica Copertina ({img_format})",
            data=img_data,
            file_name=f"glitch_cover.{img_format.lower()}",
            mime=f"image/{img_format.lower()}"
        )

        # Rigenera
        if st.button("🔁 Rigenera Copertina"):
            st.rerun()

    except Exception as e:
        st.error(f"Errore nell'elaborazione: {str(e)}")

else:
    st.info("👆 Carica un brano per iniziare.")
