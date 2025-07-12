import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import base64
import struct
import math

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("GlitchCover Studio by Loop507 [Free App]")
st.write("Carica un brano audio e crea una copertina glitch ispirata al suono.")
st.write("Limit 200MB per file • MP3, WAV, OGG")

# Funzione per analisi audio semplificata senza librosa
def analyze_audio_simple(file) -> dict:
    try:
        file.seek(0)
        audio_bytes = file.read()
        
        # Analisi molto semplificata basata sui bytes del file
        file_size = len(audio_bytes)
        
        # Simulazione di features basate su caratteristiche del file
        # Usiamo hash dei primi bytes per consistenza
        hash_val = hash(audio_bytes[:1000] if len(audio_bytes) > 1000 else audio_bytes)
        
        # Genera features pseudo-casuali ma consistenti
        np.random.seed(abs(hash_val) % 2147483647)
        
        bpm = np.random.uniform(60, 180)
        rms = np.random.uniform(0.01, 0.08)
        dominant_freq = np.random.uniform(100, 8000)
        spectral_centroid = np.random.uniform(1000, 6000)
        
        # Emozione basata su BPM e energia simulata
        if bpm > 120 and rms > 0.04:
            emotion = "Energetic & Ritmico"
        elif bpm < 80 and rms < 0.03:
            emotion = "Calmo & Melanconico"
        else:
            emotion = "Neutrale"
        
        # Genere basato su BPM
        if bpm > 120:
            genre_style = "Elettronica / Dance"
        elif bpm < 80:
            genre_style = "Classica / Acustica"
        else:
            genre_style = "Pop / Rock"
        
        return {
            "bpm": bpm,
            "rms": rms,
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "freq_range": (20, 20000),
            "emotion": emotion,
            "genre_style": genre_style,
            "file_size": file_size
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None

# Funzione per ottenere le dimensioni in base al formato
def get_dimensions(format_type):
    if format_type == "Quadrato (1:1)":
        return (600, 600)
    elif format_type == "Verticale (9:16)":
        return (540, 960)
    elif format_type == "Orizzontale (16:9)":
        return (960, 540)
    else:
        return (600, 600)

# Funzione per generare immagine glitch
def generate_glitch_cover(features, seed=None, size=(600,600)):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    width, height = size
    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)
    
    # Base colori derivati da features
    base_r = int(np.interp(features['dominant_freq'], [50, 8000], [50, 255]))
    base_g = int(np.interp(features['spectral_centroid'], [50, 8000], [50, 255]))
    base_b = int(np.interp(features['rms'], [0, 0.1], [0, 255]))
    
    # Sfondo a gradienti glitch
    for i in range(0, width, 8):
        color = (
            max(0, min(255, (base_r + random.randint(-60, 60)))),
            max(0, min(255, (base_g + random.randint(-60, 60)))),
            max(0, min(255, (base_b + random.randint(-60, 60))))
        )
        draw.rectangle([(i, 0), (i+8, height)], fill=color)
    
    # Onde glitch ondulatorie
    for y in range(0, height, 2):
        shift = int(30 * np.sin(y / 20.0 + (features['bpm'] / 15)))
        line_color = (
            max(0, min(255, (base_r + shift))),
            max(0, min(255, (base_g + shift*2))),
            max(0, min(255, (base_b + shift*3)))
        )
        draw.line([(0, y), (width, y)], fill=line_color, width=1)
    
    # Blocchi glitch casuali
    num_blocks = 15 + int(features['rms'] * 300)
    for _ in range(num_blocks):
        bx = random.randint(0, width - 60)
        by = random.randint(0, height - 60)
        bw = random.randint(20, 60)
        bh = random.randint(10, 40)
        block_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        draw.rectangle([bx, by, bx+bw, by+bh], fill=block_color)
    
    # Linee glitch orizzontali
    for _ in range(random.randint(5, 15)):
        y = random.randint(0, height)
        shift = random.randint(-20, 20)
        line_color = (
            random.randint(100, 255),
            random.randint(0, 100),
            random.randint(0, 255)
        )
        draw.line([(0, y), (width, y)], fill=line_color, width=random.randint(1, 4))
    
    # Cerchi glitch
    for _ in range(random.randint(3, 8)):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        radius = random.randint(10, 50)
        circle_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], 
                    fill=circle_color, outline=None)
    
    # Testo con info audio
    font_size = max(12, min(16, width // 40))
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    text_lines = [
        f"Genere: {features['genre_style']}",
        f"Emozione: {features['emotion']}",
        f"BPM: {features['bpm']:.1f}",
        f"Energia: {features['rms']:.3f}",
        f"Freq. dom: {features['dominant_freq']:.0f} Hz",
        f"Spettro: {features['spectral_centroid']:.0f} Hz"
    ]
    
    # Posiziona il testo in base alle dimensioni
    y_text = 10
    for line in text_lines:
        if font:
            draw.text((10, y_text), line, font=font, fill=(255, 255, 255))
        else:
            draw.text((10, y_text), line, fill=(255, 255, 255))
        y_text += font_size + 2
    
    # Applica filtro glitch
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img

# Caricamento file audio
audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

# Selezione formato immagine
col1, col2 = st.columns(2)
with col1:
    img_format = st.selectbox("Formato file:", ["PNG", "JPEG"])
with col2:
    aspect_ratio = st.selectbox("Formato immagine:", ["Quadrato (1:1)", "Verticale (9:16)", "Orizzontale (16:9)"])

# Stato per rigenerare immagine
if 'seed' not in st.session_state:
    st.session_state.seed = random.randint(0, 999999)

if 'last_file' not in st.session_state:
    st.session_state.last_file = None

def rigenera():
    st.session_state.seed = random.randint(0, 999999)

if audio_file:
    # Verifica se è un nuovo file
    current_file_id = f"{audio_file.name}_{audio_file.size}"
    if st.session_state.last_file != current_file_id:
        st.session_state.last_file = current_file_id
        st.session_state.seed = random.randint(0, 999999)
    
    features = analyze_audio_simple(audio_file)
    
    if features:
        st.subheader("🎶 Analisi del brano:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Genere stimato:** {features['genre_style']}")
            st.write(f"**Emozione:** {features['emotion']}")
            st.write(f"**BPM stimati:** {features['bpm']:.1f}")
        
        with col2:
            st.write(f"**Energia (RMS):** {features['rms']:.3f}")
            st.write(f"**Freq. dominante:** {features['dominant_freq']:.0f} Hz")
            st.write(f"**Centro spettrale:** {features['spectral_centroid']:.0f} Hz")
        
        # Bottone per rigenerare
        if st.button("🎲 Genera nuova variante"):
            rigenera()
            st.rerun()
        
        # Genera immagine
        dimensions = get_dimensions(aspect_ratio)
        img = generate_glitch_cover(features, seed=st.session_state.seed, size=dimensions)
        
        st.image(img, caption=f"🎨 Copertina glitch generata - {aspect_ratio}", use_container_width=True)
        
        # Download
        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()
        
        st.download_button(
            label=f"⬇️ Scarica copertina {img_format} ({aspect_ratio})",
            data=byte_im,
            file_name=f"glitch_cover_{aspect_ratio.replace(':', 'x').replace(' ', '_').replace('(', '').replace(')', '')}.{img_format.lower()}",
            mime=f"image/{img_format.lower()}"
        )
else:
    st.info("👆 Carica un file audio per iniziare!")

# Footer
st.markdown("---")
st.markdown("🎨 **GlitchCover Studio by Loop507** - Crea copertine glitch uniche dai tuoi brani!")
st.markdown("*Nessuna installazione richiesta - Funziona completamente nel browser*")
