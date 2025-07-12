 import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import random
import librosa

# Configurazione della pagina
st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("GlitchCover Studio by Loop507 [Free App]")
st.write("Carica un brano audio per generare la copertina glitch e informazioni audio.")

# Upload file audio
uploaded_file = st.file_uploader("🎵 Carica il tuo brano audio (max 200MB)", type=["mp3", "wav", "ogg", "flac"])

def format_float(val):
    # Sicuro: converte a float e formatta con 2 decimali
    try:
        fval = float(val)
        return f"{fval:.2f}"
    except:
        return str(val)

def generate_glitch_image(feature_value, width=512, height=512):
    """Genera immagine glitch basata su un valore numerico estratto dal brano"""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = int((feature_value * 1000) % 255)
    for y in range(height):
        for x in range(width):
            r = (base_color + (x * y) % 255) % 255
            g = (base_color * 2 + (x + y) % 255) % 255
            b = (base_color * 3 + (x - y) % 255) % 255
            # distorsione glitch casuale
            if random.random() < 0.05:
                r, g, b = b, r, g
            arr[y, x] = [r, g, b]
    return Image.fromarray(arr)

def analyze_audio(file):
    y, sr = librosa.load(file, sr=None)
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    # Frequenza dominante: max valore FFT
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    dominant_freq = freqs[np.argmax(fft)]
    return {
        "bpm": bpm,
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "dominant_freq": dominant_freq,
    }

def describe_image(bpm, rms, dominant_freq):
    return (f"Le onde distorte riflettono il ritmo elevato di {format_float(bpm)} BPM del brano, "
            f"l'intensità RMS media di {format_float(rms)} e la frequenza dominante a {format_float(dominant_freq)} Hz.")

if uploaded_file is not None:
    try:
        with st.spinner("Analisi del brano in corso..."):
            audio_data = analyze_audio(uploaded_file)
        
        st.success("Analisi completata!")

        st.write("### Dati Audio")
        st.write(f"- BPM stimato: {format_float(audio_data['bpm'])}")
        st.write(f"- Intensità RMS media: {format_float(audio_data['rms'])}")
        st.write(f"- Centro spettrale medio: {format_float(audio_data['spectral_centroid'])} Hz")
        st.write(f"- Banda spettrale media: {format_float(audio_data['spectral_bandwidth'])} Hz")
        st.write(f"- Frequenza dominante: {format_float(audio_data['dominant_freq'])} Hz")

        # Generazione immagine glitch basata su BPM, RMS, freq dominante
        st.write("### Immagine Glitch Generata")
        img = generate_glitch_image(audio_data['bpm'])
        st.image(img, caption="Copertina Glitch basata sul ritmo (BPM)", use_container_width=True)
        st.write(describe_image(audio_data['bpm'], audio_data['rms'], audio_data['dominant_freq']))

        # Pulsante download
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("⬇️ Scarica copertina PNG", byte_im, "glitch_cover.png", "image/png")

        # Pulsante rigenera (aggiorna pagina)
        if st.button("🎲 Rigenera immagine"):
            st.experimental_rerun()

    except Exception as e:
        st.error(f"Errore nell'elaborazione: {e}")
else:
    st.info("👆 Carica un file audio per iniziare!")

# Footer
st.markdown("---")
st.markdown("🎨 **GlitchCover Studio by Loop507** - Crea copertine glitch uniche per i tuoi brani.")
