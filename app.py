import streamlit as st
from pydub import AudioSegment
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import librosa

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("GlitchCover Studio by Loop507 [Free App]")
st.write("Carica un brano audio e crea una copertina glitch ispirata al suono.")
st.write("Limit 200MB per file • JPG, JPEG, PNG")

# Funzione helper per convertire numpy scalare in float Python
def to_scalar(val):
    if isinstance(val, np.ndarray):
        return float(val.item())
    return float(val)

# Caricamento file audio
audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

# Selezione formato immagine
img_format = st.selectbox("Scegli il formato immagine:", ["PNG", "JPEG"])

# Funzione per analisi audio e estrazione features base
def analyze_audio(file) -> dict:
    try:
        audio_bytes = file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # RMS energia
        rms = np.mean(librosa.feature.rms(y=y))
        # Frequenza dominante (peak nello spettro)
        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        avg_spectrum = np.mean(D, axis=1)
        dominant_freq = freqs[np.argmax(avg_spectrum)]
        # Centro spettrale
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        # Gamma frequenze
        freq_range = (freqs[0], freqs[-1])
        
        # Emozione base (semplice euristica su energia e bpm)
        if tempo > 120 and rms > 0.03:
            emotion = "Energetic & Ritmico"
        elif tempo < 70 and rms < 0.02:
            emotion = "Calmo & Melanconico"
        else:
            emotion = "Neutrale"
        
        # Genere + stile (semplice euristica)
        if tempo > 110:
            genre_style = "Elettronica / Dance"
        elif tempo < 80:
            genre_style = "Classica / Acustica"
        else:
            genre_style = "Pop / Rock"
        
        return {
            "bpm": tempo,
            "rms": rms,
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "freq_range": freq_range,
            "emotion": emotion,
            "genre_style": genre_style
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None

# Funzione per generare immagine glitch "audio-visiva" basata su feature
def generate_glitch_cover(features, seed=None, size=(600,600)):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    width, height = size
    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)
    
    # Base colori derivati da frequenze
    base_r = int(np.interp(to_scalar(features['dominant_freq']), [50, 12000], [50, 255]))
    base_g = int(np.interp(to_scalar(features['spectral_centroid']), [50, 12000], [50, 255]))
    base_b = int(np.interp(to_scalar(features['rms']), [0, 0.1], [0, 255]))
    
    # Sfondo a gradienti glitch
    for i in range(0, width, 10):
        color = (
            (base_r + random.randint(-40, 40)) % 256,
            (base_g + random.randint(-40, 40)) % 256,
            (base_b + random.randint(-40, 40)) % 256
        )
        draw.line([(i,0), (i,height)], fill=color, width=5)
    
    # Aggiunta di onde glitch ondulatorie
    for y in range(0, height, 3):
        shift = int(20 * np.sin(y / 15.0 + (features['bpm'] / 10)))
        line_color = (
            (base_r + shift) % 256,
            (base_g + shift*2) % 256,
            (base_b + shift*3) % 256
        )
        draw.line([(0,y), (width,y)], fill=line_color, width=1)
    
    # Blocchi glitch casuali
    num_blocks = 20 + int(to_scalar(features['rms']) * 500)
    for _ in range(num_blocks):
        bx = random.randint(0, width - 40)
        by = random.randint(0, height - 40)
        bw = random.randint(10, 40)
        bh = random.randint(10, 40)
        block_color = (
            random.randint(0,255),
            random.randint(0,255),
            random.randint(0,255)
        )
        draw.rectangle([bx, by, bx+bw, by+bh], fill=block_color)
    
    # Aggiunta testo descrittivo con caratteristiche
    font_size = 14
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text_lines = [
        f"Genere e Stile: {features['genre_style']}",
        f"Emozione: {features['emotion']}",
        f"BPM stimati: {to_scalar(features['bpm']):.1f}",
        f"Energia (RMS): {to_scalar(features['rms']):.4f}",
        f"Frequenza dominante: {to_scalar(features['dominant_freq']):.1f} Hz",
        f"Centro spettrale: {to_scalar(features['spectral_centroid']):.1f} Hz",
        f"Gamma frequenze: {to_scalar(features['freq_range'][0]):.1f} – {to_scalar(features['freq_range'][1]):.1f} Hz",
        "Descrizione:",
        f"Le onde distorte riflettono il ritmo di {int(to_scalar(features['bpm']))} BPM e l’energia del brano.",
    ]
    
    y_text = 10
    for line in text_lines:
        draw.text((10, y_text), line, font=font, fill=(255,255,255))
        y_text += font_size + 4
    
    # Leggera sfocatura per effetto "glitch artistico"
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    return img

# Stato per rigenerare immagine random
if 'seed' not in st.session_state:
    st.session_state.seed = random.randint(0, 999999)

def rigenera():
    st.session_state.seed = random.randint(0, 999999)

if audio_file:
    features = analyze_audio(audio_file)
    if features:
        st.subheader("🎶 Dati analizzati del brano:")
        st.write(f"Genere e Stile stimati: **{features['genre_style']}**")
        st.write(f"Emozione prevalente: **{features['emotion']}**")
        st.write(f"🎵 BPM stimati: {to_scalar(features['bpm']):.1f}")
        st.write(f"🔊 Energia (RMS): {to_scalar(features['rms']):.4f}")
        st.write(f"🔎 Frequenza dominante: {to_scalar(features['dominant_freq']):.1f} Hz")
        st.write(f"📡 Centro spettrale medio: {to_scalar(features['spectral_centroid']):.1f} Hz")
        st.write(f"🌈 Gamma frequenze: {to_scalar(features['freq_range'][0]):.1f} Hz – {to_scalar(features['freq_range'][1]):.1f} Hz")

        if st.button("🎲 Rigenera immagine"):
            rigenera()
            st.experimental_rerun()
        
        img = generate_glitch_cover(features, seed=st.session_state.seed)
        st.image(img, caption="🎨 Copertina glitch generata", use_container_width=True)
        
        # Scarica immagine
        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()
        st.download_button(
            label=f"⬇️ Scarica copertina {img_format}",
            data=byte_im,
            file_name=f"glitch_cover.{img_format.lower()}",
            mime=f"image/{img_format.lower()}"
        )
else:
    st.info("👆 Carica un file audio per iniziare!")

# Footer
st.markdown("---")
st.markdown("🎨 **GlitchCover Studio by Loop507** - Unisci arte e tecnologia per dare vita ai tuoi brani!")

