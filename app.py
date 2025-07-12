import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import random

# Config pagina
st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

# Stato per rigenerare
if "seed" not in st.session_state:
    st.session_state.seed = random.randint(0, 99999)

# Header
st.title("GlitchCover Studio by Loop507")
st.write("Carica il tuo brano per generare una copertina glitch unica e astratta, basata su analisi audio!")

# Upload audio
audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3 o WAV)", type=["mp3", "wav"])

# Formato immagine
format_option = st.selectbox("📐 Scegli il formato della copertina:", ["1:1 (Quadrato)", "4:5 (Verticale)", "16:9 (Orizzontale)"])

format_map = {
    "1:1 (Quadrato)": (800, 800),
    "4:5 (Verticale)": (800, 1000),
    "16:9 (Orizzontale)": (1280, 720)
}
img_width, img_height = format_map[format_option]

# Pulsante rigenera
if st.button("🎲 Rigenera Copertina"):
    st.session_state.seed = random.randint(0, 99999)

# Funzione generazione immagine glitch
def generate_glitch_image(width, height, seed):
    random.seed(seed)
    np.random.seed(seed)

    # Base
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Linee, blocchi e distorsioni glitch
    for _ in range(300):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = x1 + random.randint(10, 300)
        y2 = y1 + random.randint(1, 20)

        color = (
            random.randint(80, 255),
            random.randint(0, 180),
            random.randint(100, 255)
        )
        draw.rectangle([x1, y1, x2, y2], fill=color)

    # Sovrapposizione di onde/rumore
    for i in range(0, height, 5):
        shift = int(20 * np.sin(i / 30.0 + seed % 10))
        region = img.crop((0, i, width, i + 1))
        img.paste(region, (shift, i))

    return img

# Generazione e visualizzazione
if audio_file:
    st.success("Brano caricato correttamente! Analisi simulata in corso…")

    # Simula parametri audio
    bpm = random.randint(80, 150)
    freq_center = random.randint(400, 4000)
    description = f"Le onde distorte riflettono il ritmo di {bpm} BPM. Frequenze dominanti attorno ai {freq_center} Hz."

    # Crea immagine glitch
    glitch_img = generate_glitch_image(img_width, img_height, st.session_state.seed)

    st.image(glitch_img, caption=description, use_container_width=True)

    # Download
    buffer = io.BytesIO()
    glitch_img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    st.download_button("⬇️ Scarica Copertina", img_bytes, file_name="glitch_cover.png", mime="image/png")

else:
    st.info("Carica un brano per iniziare.")
