import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import librosa
import io
import random
import matplotlib.pyplot as plt

# Config pagina
st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="wide")

st.title("GlitchCover Studio by Loop507 [Free App]")
st.write("Carica un brano audio e genera una copertina glitch ispirata ai dati audio.\n"
         "Limit 200MB per file • JPG, JPEG, PNG per l’immagine finale.")

# Upload audio
uploaded_audio = st.file_uploader("📁 Carica il file audio (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

def analyze_audio(file):
    y, sr = librosa.load(file, sr=None)
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    centroid_arr = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid = np.mean(centroid_arr)
    dominant_freq_arr = librosa.piptrack(y=y, sr=sr)[0]
    # Estrai frequenza dominante media non zero
    dominant_freq = dominant_freq_arr[dominant_freq_arr > 0].mean() if np.any(dominant_freq_arr > 0) else 0
    bandwidth_arr = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth = np.mean(bandwidth_arr)
    energy = np.sum(librosa.feature.rms(y=y))
    return {
        "bpm": bpm,
        "rms": rms,
        "dominant_freq": dominant_freq,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "energy": energy
    }

def generate_glitch_cover(audio_features, width=600, height=600):
    # Crea immagine nera
    img = Image.new("RGB", (width, height), color=(0,0,0))
    draw = ImageDraw.Draw(img)

    # Parametri influenzati dai dati audio
    base_freq = audio_features["dominant_freq"] or 440  # fallback 440Hz
    bpm = audio_features["bpm"] or 120
    energy = audio_features["energy"] or 0.1
    rms = audio_features["rms"] or 0.1
    bandwidth = audio_features["bandwidth"] or 1000

    # Color palette glitch
    colors = [
        (255, 50, 50),
        (50, 255, 50),
        (50, 50, 255),
        (255, 255, 50),
        (255, 50, 255)
    ]

    # Disegna linee orizzontali distorte (glitch)
    for y in range(0, height, int(5 + 15 * rms)):
        shift = int(20 * np.sin(y / (10 + bpm/20)) * energy)
        color = random.choice(colors)
        draw.line([(shift, y), (width + shift, y)], fill=color, width=2)

    # Disegna rettangoli glitch casuali
    for _ in range(int(10 + energy * 50)):
        x0 = random.randint(0, width - 50)
        y0 = random.randint(0, height - 50)
        x1 = x0 + random.randint(10, 50)
        y1 = y0 + random.randint(5, 25)
        col = random.choice(colors)
        draw.rectangle([x0, y0, x1, y1], fill=col)

    # Aggiungi cerchi concentrici oscillanti in base a frequenze
    max_radius = int(height / 3)
    num_circles = int(5 + rms * 20)
    for i in range(num_circles):
        radius = int(max_radius * (i+1) / num_circles)
        offset = int(15 * np.sin(i * base_freq / 1000))
        bbox = [
            width//2 - radius + offset,
            height//2 - radius + offset,
            width//2 + radius + offset,
            height//2 + radius + offset
        ]
        color = colors[i % len(colors)]
        draw.ellipse(bbox, outline=color, width=2)

    return img

def describe_image(audio_features):
    bpm = int(audio_features["bpm"])
    dominant_freq = int(audio_features["dominant_freq"])
    centroid = int(audio_features["centroid"])
    bandwidth = int(audio_features["bandwidth"])

    desc = (
        f"Questa copertina mostra onde distorte e linee glitchate ispirate a un brano con {bpm} BPM.\n"
        f"La frequenza dominante è di circa {dominant_freq} Hz, mentre il centro spettrale è a {centroid} Hz.\n"
        f"La gamma di frequenze del brano si estende approssimativamente da {max(0, centroid - bandwidth//2)} Hz "
        f"a {centroid + bandwidth//2} Hz, creando un effetto visivo dinamico e vibrante."
    )
    return desc

if uploaded_audio is not None:
    with st.spinner("Analisi in corso..."):
        try:
            audio_features = analyze_audio(uploaded_audio)
            st.success("Analisi completata!")

            st.write("### Dati tecnici del brano:")
            st.write(f"- BPM stimati: {audio_features['bpm']:.2f}")
            st.write(f"- Energia (RMS): {audio_features['rms']:.4f}")
            st.write(f"- Frequenza dominante: {audio_features['dominant_freq']:.2f} Hz")
            st.write(f"- Centro spettrale medio: {audio_features['centroid']:.2f} Hz")
            st.write(f"- Gamma frequenze (bandwidth): {audio_features['bandwidth']:.2f} Hz")

            st.write("---")

            # Genera immagine
            img = generate_glitch_cover(audio_features)
            st.image(img, caption="🎨 Copertina glitch generata", use_container_width=False)

            # Descrizione
            desc = describe_image(audio_features)
            st.write("### Descrizione della copertina:")
            st.info(desc)

            # Download immagine
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="⬇️ Scarica la copertina",
                data=byte_im,
                file_name="glitch_cover.png",
                mime="image/png"
            )

            # Pulsante rigenera (rinfresca la pagina)
            if st.button("🔄 Rigenera la copertina"):
                st.experimental_rerun()

        except Exception as e:
            st.error(f"Errore durante l'analisi o la generazione: {e}")

else:
    st.info("Carica un file audio per iniziare.")

# Footer
st.markdown("---")
st.markdown("🎨 **GlitchCover Studio by Loop507** — Arte e tecnologia per copertine musicali uniche.")
