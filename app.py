import streamlit as st
from PIL import Image
import numpy as np
import io
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Funzione per analisi audio
def analyze_audio(file):
    y, sr = librosa.load(file, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    # Frequenza dominante: freq con intensità massima nello spettro medio
    avg_spectrum = np.mean(S, axis=1)
    dominant_freq = freqs[np.argmax(avg_spectrum)]
    return {
        "bpm": round(tempo),
        "rms": round(rms, 5),
        "spectral_centroid": round(spectral_centroid),
        "dominant_freq": round(dominant_freq),
        "duration": round(librosa.get_duration(y=y, sr=sr), 2)
    }

# Funzione per creare immagine glitch ispirata ai dati audio
def create_glitch_image(features, width, height, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    bpm = features["bpm"]
    rms = features["rms"]
    centroid = features["spectral_centroid"]
    dom_freq = features["dominant_freq"]

    # Parametri immagine influenzati da dati audio
    base_color = int(min(255, bpm * 2))
    noise_level = int(min(50, rms * 1000))
    wave_freq = centroid / 1000
    shift_intensity = int(min(50, dom_freq / 10))

    arr = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        shift = int(shift_intensity * np.sin(y * wave_freq))
        for x in range(width):
            val = (base_color + noise_level * np.random.randn()) % 256
            r = int((val + shift) % 256)
            g = int((val + 2*shift) % 256)
            b = int((val + 3*shift) % 256)
            arr[y, (x + shift) % width] = [r, g, b]

    img = Image.fromarray(arr, 'RGB')
    return img

# Interfaccia Streamlit
st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("GlitchCover Studio by Loop507 [Free App]")
st.write("Carica un file audio e genera una copertina glitch ispirata al tuo brano.")
st.write("**Limit 200MB per file • Formati supportati: mp3, wav, flac, ogg**")

audio_file = st.file_uploader("Carica il tuo brano audio", type=["mp3", "wav", "flac", "ogg"])

format_option = st.selectbox("Scegli il formato immagine", options=["Quadrato 1:1", "Orizzontale 16:9", "Verticale 9:16"])

if audio_file is not None:
    try:
        features = analyze_audio(audio_file)
        st.subheader("Analisi audio")
        st.write(f"**Durata:** {features['duration']} secondi")
        st.write(f"**BPM (tempo):** {features['bpm']}")
        st.write(f"**Energia (RMS):** {features['rms']}")
        st.write(f"**Centro spettrale medio:** {features['spectral_centroid']} Hz")
        st.write(f"**Frequenza dominante:** {features['dominant_freq']} Hz")

        # Dimensioni immagine in base al formato scelto
        if format_option == "Quadrato 1:1":
            width, height = 512, 512
        elif format_option == "Orizzontale 16:9":
            width, height = 768, 432
        else:
            width, height = 432, 768

        # Bottone rigenera immagine
        if "seed" not in st.session_state:
            st.session_state.seed = random.randint(0, 10000)

        if st.button("🎲 Rigenera immagine"):
            st.session_state.seed = random.randint(0, 10000)

        glitch_img = create_glitch_image(features, width, height, seed=st.session_state.seed)
        st.image(glitch_img, caption="Copertina Glitch generata", use_container_width=True)

        # Descrizione immagine
        description = (
            f"L'immagine riflette il ritmo di {features['bpm']} BPM, "
            f"con colori influenzati dall'energia (RMS={features['rms']}) e distorsioni "
            f"basate sulla frequenza dominante di {features['dominant_freq']} Hz."
        )
        st.markdown(f"**Descrizione immagine:** {description}")

        # Download immagine
        buf = io.BytesIO()
        glitch_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("⬇️ Scarica copertina PNG", data=byte_im, file_name="glitch_cover.png", mime="image/png")

    except Exception as e:
        st.error(f"Errore nell'elaborazione: {str(e)}")

else:
    st.info("👆 Carica un file audio per iniziare")

# Footer
st.markdown("---")
st.markdown("🎨 **GlitchCover Studio by Loop507** - La tua arte audio-visiva personale")
