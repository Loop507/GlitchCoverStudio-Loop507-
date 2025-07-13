import streamlit as st
import numpy as np
import io
import random
from PIL import Image, ImageDraw, ImageChops
import hashlib
import time
import librosa

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")

st.title("🎧 GlitchCover Studio by Loop507 [Free App]")
st.markdown("Crea una copertina glitch unica basata sul tuo brano. Carica un file audio e genera arte ispirata al suono.")

def get_dimensions(format_type):
    if format_type == "Quadrato (1:1)":
        return (800, 800)
    elif format_type == "Verticale (9:16)":
        return (720, 1280)
    elif format_type == "Orizzontale (16:9)":
        return (1280, 720)
    else:
        return (800, 800)

def analyze_audio_librosa(file):
    try:
        file.seek(0)
        audio_bytes = file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=30)

        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(bpm) if bpm is not None else 60.0

        rms = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        dynamic_range = spectral_bandwidth - spectral_centroid

        hash_obj = hashlib.sha256(audio_bytes[:10000]).hexdigest()
        hash_int = int(hash_obj[:8], 16)

        if rms > 0.05 and bpm > 110:
            emotion = "Energetico"
        elif rms < 0.02 and bpm < 80:
            emotion = "Calmo"
        elif dynamic_range > 2000:
            emotion = "Dinamico"
        else:
            emotion = "Equilibrato"

        return {
            "bpm": bpm,
            "rms": rms,
            "spectral_centroid": spectral_centroid,
            "dynamic_range": dynamic_range,
            "emotion": emotion,
            "file_hash": hash_int,
            "file_size": len(audio_bytes)
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio con Librosa: {str(e)}")
        return None

def apply_glitch_effect(img, seed):
    random.seed(seed)
    img = img.convert("RGB")
    width, height = img.size

    # GLITCH PIÙ AGGRESSIVO E VARIABILE
    shift_intensity = random.randint(5, 25)  # Shift più variabile
    
    r, g, b = img.split()
    r = ImageChops.offset(r, random.randint(-shift_intensity, shift_intensity), random.randint(-shift_intensity, shift_intensity))
    g = ImageChops.offset(g, random.randint(-shift_intensity, shift_intensity), random.randint(-shift_intensity, shift_intensity))
    b = ImageChops.offset(b, random.randint(-shift_intensity, shift_intensity), random.randint(-shift_intensity, shift_intensity))
    img = Image.merge("RGB", (r, g, b))

    draw = ImageDraw.Draw(img)
    
    # Rumore più intenso
    noise_points = random.randint(100, 400)
    for _ in range(noise_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = tuple(random.randint(0, 255) for _ in range(3))
        # Varia la dimensione del rumore
        size = random.randint(1, 3)
        draw.rectangle([x, y, x + size, y + size], fill=color)

    # Linee di disturbo più varie
    line_spacing = random.randint(2, 8)
    line_alpha = random.randint(10, 80)
    for y in range(0, height, line_spacing):
        if random.random() < 0.7:  # Non tutte le linee
            line_color = (random.randint(0, line_alpha), random.randint(0, line_alpha), random.randint(0, line_alpha))
            draw.line([(0, y), (width, y)], fill=line_color, width=random.randint(1, 2))

    # Effetto pixelation variabile
    pixelation_factor = random.randint(8, 25)
    small = img.resize((width // pixelation_factor, height // pixelation_factor), Image.BILINEAR)
    img = small.resize(img.size, Image.NEAREST)

    # AGGIUNTA: Disturbo a blocchi casuali
    for _ in range(random.randint(3, 10)):
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = x1 + random.randint(50, 200)
        y2 = y1 + random.randint(50, 200)
        
        # Copia una sezione e spostala
        if x2 < width and y2 < height:
            section = img.crop((x1, y1, x2, y2))
            new_x = random.randint(0, width - (x2 - x1))
            new_y = random.randint(0, height - (y2 - y1))
            img.paste(section, (new_x, new_y))

    return img

def generate_glitch_image(features, seed, size=(800, 800)):
    # FORZA CASUALITÀ MAGGIORE
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    
    # Aggiungi variazione extra al seed per ogni elemento
    variation_seed = seed + random.randint(1, 10000)
    
    width, height = size

    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)

    palettes = {
        "Energetico": [(255, 70, 30), (255, 140, 0), (255, 90, 60)],
        "Calmo": [(30, 90, 160), (60, 120, 130), (0, 160, 130)],
        "Dinamico": [(255, 0, 120), (0, 255, 70), (60, 90, 255), (255, 255, 0)],
        "Equilibrato": [(130, 0, 180), (0, 160, 160), (160, 120, 255)]
    }
    colors = palettes.get(features["emotion"], palettes["Equilibrato"])

    # VARIAZIONE MAGGIORE NEI BLOCCHI
    block_size = max(5, int(features["rms"] * 300) + random.randint(-20, 20))
    block_offset = random.randint(0, block_size // 2)  # Offset casuale per i blocchi
    
    for x in range(block_offset, width, block_size):
        for y in range(block_offset, height, block_size):
            # Più variazione nei colori e dimensioni
            jitter = random.randint(-80, 80)
            size_jitter = random.randint(-5, 10)
            actual_size = max(1, block_size + size_jitter)
            
            base_color = random.choice(colors)
            color = tuple(max(0, min(255, c + jitter)) for c in base_color)
            draw.rectangle([x, y, x + actual_size, y + actual_size], fill=color)

    # LINEE PIÙ CASUALI
    line_count = int(features["dynamic_range"] // 500) + random.randint(5, 15)
    for _ in range(line_count):
        # Linee in direzioni più varie
        if random.random() < 0.3:  # Linee orizzontali
            start_x, start_y = 0, random.randint(0, height)
            end_x, end_y = width, random.randint(0, height)
        elif random.random() < 0.6:  # Linee verticali
            start_x, start_y = random.randint(0, width), 0
            end_x, end_y = random.randint(0, width), height
        else:  # Linee diagonali
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            end_x = random.randint(0, width)
            end_y = random.randint(0, height)
            
        draw.line((start_x, start_y, end_x, end_y), fill=random.choice(colors), width=random.randint(1, 6))

    # CERCHI PIÙ VARI
    circle_count = int(features["bpm"] // 5) + random.randint(2, 8)
    for _ in range(circle_count):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        r = int(features["rms"] * 150) + random.randint(5, 50)
        
        # Varia tra cerchi pieni e vuoti
        if random.random() < 0.5:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=random.choice(colors), width=random.randint(1, 5))
        else:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=random.choice(colors))

    # AGGIUNGI ELEMENTI CASUALI EXTRA
    for _ in range(random.randint(10, 30)):
        shape_type = random.choice(['rect', 'line', 'point'])
        color = random.choice(colors)
        
        if shape_type == 'rect':
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = x1 + random.randint(10, 100), y1 + random.randint(10, 100)
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape_type == 'line':
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line((x1, y1, x2, y2), fill=color, width=random.randint(1, 3))
        else:  # point
            for _ in range(random.randint(5, 20)):
                x, y = random.randint(0, width), random.randint(0, height)
                draw.point((x, y), fill=color)

    # Effetto glitch finale con seed diverso
    img = apply_glitch_effect(img, variation_seed)

    descrizione = [
        f"🎵 BPM stimato: {features['bpm']:.1f}",
        f"🔊 RMS (energia media): {features['rms']:.3f}",
        f"📡 Centro spettrale: {features['spectral_centroid']:.0f} Hz",
        f"🎭 Stile emotivo: {features['emotion']}"
    ]

    return img, descrizione

# --- UI ---

if 'regen_count' not in st.session_state:
    st.session_state.regen_count = 0

audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

col1, col2, col3 = st.columns(3)
with col1:
    img_format = st.selectbox("Formato immagine:", ["PNG", "JPEG"])
with col2:
    aspect_ratio = st.selectbox("Formato dimensioni:", ["Quadrato (1:1)", "Verticale (9:16)", "Orizzontale (16:9)"])
with col3:
    show_analysis = st.checkbox("Mostra analisi", value=True)

if audio_file:
    features = analyze_audio_librosa(audio_file)
    if features:
        # MOSTRA L'ANALISI SUBITO DOPO L'ELABORAZIONE
        if show_analysis:
            st.markdown("### 🎨 Analisi Audio")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🎵 BPM", f"{features['bpm']:.1f}")
                st.metric("🔊 RMS", f"{features['rms']:.3f}")
            with col_b:
                st.metric("📡 Centro spettrale", f"{features['spectral_centroid']:.0f} Hz")
                st.metric("🎭 Emozione", features['emotion'])
        
        dimensions = get_dimensions(aspect_ratio)
        
        # MIGLIORA IL SEED PER VARIAZIONI REALI
        base_seed = features['file_hash']
        
        if st.button("🔄 Rigenera Copertina"):
            st.session_state.regen_count += 1
            st.rerun()  # Forza il refresh della pagina

        # CREA UN SEED COMPLETAMENTE DIVERSO AD OGNI RIGENERAZIONE
        if st.session_state.regen_count == 0:
            seed = base_seed  # Prima volta usa il seed del file
        else:
            # Rigenerazioni successive: seed completamente casuali
            seed = int(time.time() * 1000000) + random.randint(1, 999999) + (st.session_state.regen_count * 54321)

        with st.spinner("🎨 Creazione copertina glitch..."):
            img, descrizione = generate_glitch_image(features, seed, size=dimensions)

        st.image(img, caption=f"Copertina glitch generata - {aspect_ratio} (Versione #{st.session_state.regen_count + 1})", use_container_width=True)

        # MOSTRA I DETTAGLI TECNICI DELLA GENERAZIONE
        if show_analysis:
            with st.expander("🔧 Dettagli Tecnici Generazione"):
                st.write(f"**Seed utilizzato:** {seed}")
                st.write(f"**Numero rigenerazioni:** {st.session_state.regen_count + 1}")
                st.write("**Parametri utilizzati:**")
                for d in descrizione:
                    st.markdown(f"- {d}")

        buf = io.BytesIO()
        img.save(buf, format=img_format)
        byte_im = buf.getvalue()

        filename = f"glitch_cover_{features['emotion'].lower()}_{int(time.time())}.{img_format.lower()}"
        st.download_button(
            label=f"⬇️ Scarica {img_format} ({dimensions[0]}x{dimensions[1]})",
            data=byte_im,
            file_name=filename,
            mime=f"image/{img_format.lower()}"
        )
else:
    st.session_state.regen_count = 0
    st.info("👆 Carica un file audio per iniziare!")
